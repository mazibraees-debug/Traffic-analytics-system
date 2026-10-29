"""Video -> detections/tracks -> analytics -> SQLite."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Optional

from . import db
from .analytics import FrameAnalyzer, ref_point

LEVEL_COLORS = {"free": (80, 175, 76), "moderate": (0, 200, 255),
                "heavy": (0, 140, 255), "congested": (60, 60, 230)}


def draw_overlay(frame, tracks, analyzer: FrameAnalyzer, totals: dict):
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    cfg = analyzer.cfg
    poly = analyzer.roi_poly
    if poly:
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 200, 0))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 1)
    for ln in cfg.get("lines", []):
        a = (int(ln["p1"][0] * w), int(ln["p1"][1] * h))
        b = (int(ln["p2"][0] * w), int(ln["p2"][1] * h))
        cv2.line(frame, a, b, (0, 215, 255), 2)
        cv2.putText(frame, ln["name"], (a[0] + 4, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)
    for t in tracks:
        x1, y1, x2, y2 = (int(t.bbox[0] * w), int(t.bbox[1] * h), int(t.bbox[2] * w), int(t.bbox[3] * h))
        bad = t.id in analyzer.wrong_way.flagged
        col = (0, 0, 255) if bad else (60, 220, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2 if bad else 1)
        cv2.putText(frame, f"{'WRONG WAY ' if bad else ''}{t.cls} #{t.id}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        px, py = ref_point(t.bbox, analyzer.ref_mode)
        cv2.circle(frame, (int(px * w), int(py * h)), 3, col, -1)
    live = analyzer.live
    cv2.rectangle(frame, (0, 0), (w, 24), (30, 30, 30), -1)
    lv = live["level"]
    cv2.circle(frame, (14, 12), 6, LEVEL_COLORS[lv], -1)
    txt = f"{lv.upper()}  in-ROI {live['count']}  counted {sum(totals.values())}"
    cv2.putText(frame, txt, (28, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return frame


def run_analysis(cfg: dict, video_path, db_path, run_id: int, start_time: datetime, detector,
                 stop_event=None, on_progress: Optional[Callable] = None,
                 on_preview: Optional[Callable] = None, max_frames: Optional[int] = None) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vcfg = cfg["video"]
    stride = max(1, round(src_fps / float(vcfg.get("target_fps", 8))))
    width = int(vcfg.get("process_width", 640))
    preview_every = int(vcfg.get("preview_every", 2))
    quality = int(vcfg.get("preview_quality", 70))

    analyzer = FrameAnalyzer(cfg["analytics"])
    totals: dict[str, int] = {}
    cross_buf, dens_buf, evt_buf = [], [], []

    def flush():
        db.insert_batch(db_path, run_id, start_time, cross_buf, dens_buf, evt_buf)
        cross_buf.clear(); dens_buf.clear(); evt_buf.clear()

    t0 = time.time()
    idx, processed, vt = -1, 0, 0.0
    try:
        while not (stop_event and stop_event.is_set()):
            if not cap.grab():
                break
            idx += 1
            if idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            vt = idx / src_fps
            tracks = detector.track(frame)
            res = analyzer.update(tracks, vt)
            for c in res["crossings"]:
                totals[c["vclass"]] = totals.get(c["vclass"], 0) + 1
            cross_buf += res["crossings"]; dens_buf += res["density"]; evt_buf += res["events"]
            processed += 1
            if processed % 50 == 0:
                flush()
            if on_preview and processed % preview_every == 0:
                img = draw_overlay(frame, tracks, analyzer, totals)
                ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok2:
                    on_preview(buf.tobytes())
            if on_progress:
                on_progress(processed, (idx / total) if total else 0.0, vt)
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        dens_buf += analyzer.finish()
        flush()
    return {"frames": processed, "video_seconds": vt, "elapsed_s": time.time() - t0,
            "totals": totals, "stopped": bool(stop_event and stop_event.is_set())}
