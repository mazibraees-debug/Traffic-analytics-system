"""Populate the database with SIMULATED traffic so you can try the dashboard
without a video or a model. It drives the real analytics engine
(line counting, congestion, wrong-way) with synthetic vehicles.

    python scripts/seed_demo.py --hours 3 --start 2026-10-29T07:00:00
"""
import argparse
import math
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db  # noqa: E402
from app.analytics import FrameAnalyzer, Track  # noqa: E402
from app.config import load_config, resolve  # noqa: E402

CLASS_MIX = [("car", 0.72), ("motorcycle", 0.10), ("truck", 0.10), ("bus", 0.08)]
SIZE = {"car": (0.07, 0.09), "motorcycle": (0.03, 0.05), "truck": (0.10, 0.14), "bus": (0.10, 0.15)}
DOWN_LANES = [0.18, 0.30, 0.42]   # expected direction: down the screen
UP_LANES = [0.58, 0.70, 0.82]     # opposite carriageway: up the screen


def pick_class(rng):
    r, acc = rng.random(), 0
    for c, p in CLASS_MIX:
        acc += p
        if r <= acc:
            return c
    return "car"


def arrival_rate(hour_of_day: float) -> float:
    """Vehicles / second / direction: morning and evening rush peaks."""
    return 0.10 + 0.35 * math.exp(-((hour_of_day - 8.5) / 0.7) ** 2) \
        + 0.6 * math.exp(-((hour_of_day - 17.5) / 0.9) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--start", default="2026-10-29T07:00:00")
    ap.add_argument("--fps", type=float, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cfg = load_config()
    dbp = resolve(cfg["paths"]["database"])
    db.init_db(dbp)
    start = datetime.fromisoformat(a.start)
    run_id = db.create_run(dbp, "demo (simulated)", start)
    an = FrameAnalyzer(cfg["analytics"])

    dt = 1.0 / a.fps
    steps = int(a.hours * 3600 * a.fps)
    vehicles, next_id = [], 1
    cross, dens, evts = [], [], []
    base_speed = 0.22
    for step in range(steps):
        t = step * dt
        hod = start.hour + start.minute / 60 + t / 3600
        rate = arrival_rate(hod)
        n_now = len(vehicles)
        slow = max(0.12, 1 - n_now / 22)
        if 8.58 <= hod <= 8.65:      # simulated incident: traffic crawls for ~4 minutes
            slow *= 0.25
        for direction in (+1, -1):
            if rng.random() < rate * dt and n_now < 40:
                wrong = rng.random() < 0.0015
                lane_dir = -direction if wrong else direction
                lanes = DOWN_LANES if lane_dir == +1 else UP_LANES
                cls = pick_class(rng)
                w, h = SIZE[cls]
                x = rng.choice(lanes) + rng.uniform(-0.02, 0.02)
                # vehicles travelling "down" start at the top; "up" at the bottom
                y = 0.18 if direction == +1 else 1.05
                vehicles.append({"id": next_id, "cls": cls, "x": x, "y": y, "dir": direction,
                                 "v": base_speed * rng.uniform(0.8, 1.2), "w": w, "h": h})
                next_id += 1
        for v in vehicles:
            v["y"] += v["dir"] * v["v"] * slow * dt
            v["x"] += rng.uniform(-0.002, 0.002)
        vehicles = [v for v in vehicles if 0.12 < v["y"] < 1.08]
        tracks = [Track(v["id"], v["cls"], (v["x"] - v["w"] / 2, v["y"] - v["h"], v["x"] + v["w"] / 2, v["y"]))
                  for v in vehicles if 0 <= v["y"] <= 1.05]
        r = an.update(tracks, t)
        cross += r["crossings"]; dens += r["density"]; evts += r["events"]
        if len(cross) + len(dens) > 2000:
            db.insert_batch(dbp, run_id, start, cross, dens, evts)
            cross, dens, evts = [], [], []
    dens += an.finish()
    db.insert_batch(dbp, run_id, start, cross, dens, evts)
    db.finish_run(dbp, run_id, "done", steps, steps * dt)
    t = db.get_totals(dbp, run_id)
    print(f"Seeded run #{run_id}: {t['total_vehicles']} vehicles counted, "
          f"{t['wrong_way_events']} wrong-way events, "
          f"{t['heavy_pct']}% heavy / {t['congested_pct']}% congested samples")


if __name__ == "__main__":
    main()
