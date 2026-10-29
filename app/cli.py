"""Command line: python -m app.cli path/to/video.mp4 [--start-time 2026-10-29T08:00:00]"""
import argparse
from datetime import datetime
from pathlib import Path

from . import db, reports
from .config import load_config, resolve
from .detector import load_detector
from .pipeline import run_analysis


def main():
    ap = argparse.ArgumentParser(description="Analyse a traffic video and store results in SQLite.")
    ap.add_argument("video")
    ap.add_argument("--config", default=None)
    ap.add_argument("--start-time", default=None, help="Wall-clock time of the first frame (ISO). Default: now")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dbp = resolve(cfg["paths"]["database"])
    db.init_db(dbp)
    start = datetime.fromisoformat(a.start_time) if a.start_time else datetime.now().replace(microsecond=0)
    run_id = db.create_run(dbp, Path(a.video).name, start)
    detector = load_detector(cfg["detector"])

    def prog(n, frac, vt):
        if n % 25 == 0:
            print(f"\r  frame {n}  video {vt:7.1f}s  {frac * 100:5.1f}%", end="", flush=True)

    try:
        s = run_analysis(cfg, a.video, dbp, run_id, start, detector, on_progress=prog, max_frames=a.max_frames)
        db.finish_run(dbp, run_id, "done", s["frames"], s["video_seconds"])
    except Exception as e:
        db.finish_run(dbp, run_id, "error", message=str(e))
        raise
    print(f"\nRun #{run_id}: {s['frames']} frames in {s['elapsed_s']:.1f}s -> {s['totals']}")
    print(reports.to_csv(reports.hourly_report(dbp, run_id)))


if __name__ == "__main__":
    main()
