import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from app import db  # noqa: E402
from app.analytics import Track  # noqa: E402
from app.config import ROOT, load_config  # noqa: E402
from app.pipeline import run_analysis  # noqa: E402


class FakeDetector:
    """Stands in for YOLO+ByteTrack: one car drives down the left lane, one truck up the right lane."""
    def __init__(self):
        self.n = 0
        self.frame_shapes = []

    def track(self, frame):
        self.frame_shapes.append(frame.shape[:2])
        k = self.n
        self.n += 1
        y_down = 0.2 + 0.05 * k
        y_up = 0.98 - 0.05 * k
        out = []
        if y_down < 1.0:
            out.append(Track(1, "car", (0.2, y_down - 0.08, 0.28, y_down)))
        if y_up > 0.2:
            out.append(Track(2, "truck", (0.7, y_up - 0.12, 0.8, y_up)))
        return out


def test_pipeline_end_to_end(tmp_path):
    vid = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    for _ in range(24 * 12):
        w.write(np.zeros((180, 320, 3), np.uint8))
    w.release()

    cfg = load_config(ROOT / "config.yaml")
    cfg["video"]["target_fps"] = 8
    dbp = tmp_path / "p.db"
    db.init_db(dbp)
    start = datetime(2026, 10, 29, 8, 0, 0)
    rid = db.create_run(dbp, "v.mp4", start)
    previews = []
    detector = FakeDetector()
    s = run_analysis(cfg, vid, dbp, rid, start, detector, on_preview=previews.append)
    assert s["frames"] >= 90                      # 12 s at ~8 fps
    assert set(detector.frame_shapes) == {(180, 320)}  # do not upscale smaller sources
    t = db.get_totals(dbp, rid)
    assert t["total_vehicles"] == 2
    assert t["by_class"] == {"car": 1, "truck": 1}
    assert {(x["direction"]) for x in t["by_line"]} == {"down", "up"}
    assert t["wrong_way_events"] == 0             # both travel with their carriageway
    assert previews and previews[0][:2] == b"\xff\xd8"
    assert db.get_density(dbp, rid)
