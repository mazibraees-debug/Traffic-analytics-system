import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analytics import (FrameAnalyzer, Track, classify_level, point_in_polygon,  # noqa: E402
                           segments_cross)

CFG = {
    "reference_point": "bottom_center", "sample_interval_s": 2,
    "lines": [{"name": "L", "p1": [0, 0.5], "p2": [1, 0.5], "positive_label": "down", "negative_label": "up"}],
    "roi": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "capacity": 10, "min_vehicles": 3,
            "moderate_ratio": 0.25, "heavy_ratio": 0.5, "congested_ratio": 0.7, "slow_speed": 0.06},
    "wrong_way": {"enabled": True, "history": 6, "min_displacement": 0.05, "confirm_frames": 3,
                  "angle_threshold_deg": 120,
                  "zones": [{"name": "Main", "polygon": None, "expected_direction": [0, 1]}]},
}


def car(i, x, y, cls="car"):
    return Track(i, cls, (x - 0.02, y - 0.04, x + 0.02, y))


def drive(an, tid, x, y0, y1, steps, t0=0.0, dt=0.125):
    out = {"crossings": [], "events": [], "density": []}
    for k in range(steps + 1):
        y = y0 + (y1 - y0) * k / steps
        r = an.update([car(tid, x, y)], t0 + k * dt)
        for key in out:
            out[key] += r[key]
    return out


def test_geometry():
    assert segments_cross((0.5, 0.4), (0.5, 0.6), (0, 0.5), (1, 0.5))
    assert not segments_cross((0.5, 0.1), (0.5, 0.3), (0, 0.5), (1, 0.5))
    assert point_in_polygon((0.5, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not point_in_polygon((1.5, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)])


def test_line_counted_once_with_direction():
    an = FrameAnalyzer(CFG)
    down = drive(an, 1, 0.3, 0.2, 0.9, 20)
    assert [(c["direction"], c["vclass"]) for c in down["crossings"]] == [("down", "car")]
    up = drive(an, 2, 0.7, 0.9, 0.2, 20)
    assert [c["direction"] for c in up["crossings"]] == [("up")]
    # jitter back and forth must not double count
    an2 = FrameAnalyzer(CFG)
    total = 0
    for k, y in enumerate([0.45, 0.55, 0.45, 0.55, 0.45]):
        total += len(an2.update([car(1, 0.5, y)], k * 0.1)["crossings"])
    assert total == 1


def test_wrong_way_detected_only_against_flow():
    an = FrameAnalyzer(CFG)
    ok = drive(an, 1, 0.3, 0.1, 0.95, 25)
    assert ok["events"] == []
    bad = drive(an, 2, 0.6, 0.95, 0.1, 25)
    assert len(bad["events"]) == 1 and bad["events"][0]["type"] == "wrong_way"
    # reported once per vehicle
    more = drive(an, 2, 0.6, 0.5, 0.1, 10, t0=10)
    assert more["events"] == []


def test_stationary_vehicle_not_wrong_way():
    an = FrameAnalyzer(CFG)
    ev = []
    for k in range(30):
        ev += an.update([car(1, 0.5, 0.7 + (0.002 if k % 2 else 0))], k * 0.125)["events"]
    assert ev == []


def test_congestion_levels():
    roi = CFG["roi"]
    assert classify_level(1, 0.2, roi)[0] == "free"
    assert classify_level(3, 0.2, roi)[0] == "moderate"
    assert classify_level(6, 0.2, roi)[0] == "heavy"
    assert classify_level(8, 0.02, roi)[0] == "congested"
    assert classify_level(8, 0.30, roi)[0] == "heavy"      # dense but flowing


def test_density_samples_emitted():
    an = FrameAnalyzer(CFG)
    samples = []
    for k in range(40):  # 5 s of video at 8 fps, 8 stopped vehicles -> congested
        tracks = [car(i, 0.1 + 0.1 * i, 0.8) for i in range(8)]
        samples += an.update(tracks, k * 0.125)["density"]
    samples += an.finish()
    assert samples and samples[-1]["level"] in ("heavy", "congested")
    assert samples[0]["vehicles"] == 8
