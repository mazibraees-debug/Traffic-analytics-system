"""Draw the counting lines, ROI and wrong-way zones from config.yaml on a frame
of your video so you can calibrate them.

    python scripts/preview_config.py data/videos/my_clip.mp4 [--frame 100]
Writes data/calibration.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config()
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, f = cap.read()
    if not ok:
        sys.exit("Could not read a frame from the video")
    w0 = cfg["video"]["process_width"]
    h, w = f.shape[:2]
    f = cv2.resize(f, (w0, int(h * w0 / w)))
    h, w = f.shape[:2]
    an = cfg["analytics"]

    def px(p):
        return int(p[0] * w), int(p[1] * h)

    if an["roi"].get("polygon"):
        cv2.polylines(f, [np.array([px(p) for p in an["roi"]["polygon"]], np.int32)], True, (255, 200, 0), 2)
    for ln in an["lines"]:
        cv2.line(f, px(ln["p1"]), px(ln["p2"]), (0, 215, 255), 2)
        cv2.putText(f, ln["name"], px(ln["p1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
    for z in an["wrong_way"].get("zones", []):
        poly = z.get("polygon") or [[0, 0], [1, 0], [1, 1], [0, 1]]
        pts = [px(p) for p in poly]
        cv2.polylines(f, [np.array(pts, np.int32)], True, (0, 120, 255), 1)
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        dx, dy = z["expected_direction"]
        cv2.arrowedLine(f, (cx, cy), (int(cx + dx * 60), int(cy + dy * 60)), (0, 120, 255), 3)
        cv2.putText(f, z["name"], (cx - 60, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1)
    out = ROOT / "data" / "calibration.jpg"
    cv2.imwrite(str(out), f)
    print("Wrote", out)


if __name__ == "__main__":
    main()
