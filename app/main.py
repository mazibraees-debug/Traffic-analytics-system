"""FastAPI server: run analyses, query results, serve the dashboard."""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, reports
from .config import load_config, resolve

CFG = load_config()
DB = resolve(CFG["paths"]["database"])
VIDEOS = resolve(CFG["paths"]["videos"])
VIDEOS.mkdir(parents=True, exist_ok=True)
db.init_db(DB)
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

app = FastAPI(title="Intelligent Traffic Analytics", version="1.0")


class Job:
    def __init__(self):
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.preview: Optional[bytes] = None
        self.state = {"status": "idle", "run_id": None, "progress": 0.0,
                      "frames": 0, "video_time": 0.0, "message": ""}

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()


JOB = Job()


def _worker(video: Path, run_id: int, start: datetime, max_frames: Optional[int]):
    from .detector import load_detector
    from .pipeline import run_analysis
    st = JOB.state
    try:
        st["message"] = "Loading model..."
        detector = load_detector(CFG["detector"])
        st["message"] = "Analysing video"

        def prog(n, frac, vt):
            st.update(frames=n, progress=round(frac, 4), video_time=round(vt, 1))

        def prev(b):
            JOB.preview = b

        s = run_analysis(CFG, video, DB, run_id, start, detector, JOB.stop, prog, prev, max_frames)
        status = "stopped" if s["stopped"] else "done"
        db.finish_run(DB, run_id, status, s["frames"], s["video_seconds"])
        st.update(status=status, progress=1.0 if status == "done" else st["progress"], message="")
    except Exception as e:  # noqa: BLE001
        db.finish_run(DB, run_id, "error", message=str(e))
        st.update(status="error", message=str(e))


class ProcessRequest(BaseModel):
    video: str
    start_time: Optional[str] = None
    max_frames: Optional[int] = None


def _run(run_id: Optional[int]) -> Optional[int]:
    return run_id if run_id is not None else db.latest_run_id(DB)


# ------------------------------------------------------------------ control
@app.get("/api/config")
def get_config():
    a = CFG["analytics"]
    return {"lines": a.get("lines", []), "roi": a.get("roi", {}), "classes": list(CFG["detector"]["classes"].values())}


@app.get("/api/videos")
def list_videos():
    return sorted(p.name for p in VIDEOS.iterdir() if p.suffix.lower() in EXTS)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in EXTS:
        raise HTTPException(400, f"Unsupported file type. Use one of {sorted(EXTS)}")
    dest = VIDEOS / name
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"video": name}


@app.post("/api/process")
def process(req: ProcessRequest):
    if JOB.running:
        raise HTTPException(409, "An analysis is already running")
    video = VIDEOS / Path(req.video).name
    if not video.exists():
        raise HTTPException(404, f"Video not found in {VIDEOS}")
    try:
        start = datetime.fromisoformat(req.start_time) if req.start_time else datetime.now().replace(microsecond=0)
    except ValueError:
        raise HTTPException(400, "start_time must be ISO format, e.g. 2026-10-29T08:00:00")
    run_id = db.create_run(DB, video.name, start)
    JOB.stop.clear()
    JOB.preview = None
    JOB.state.update(status="running", run_id=run_id, progress=0.0, frames=0, video_time=0.0, message="Starting")
    JOB.thread = threading.Thread(target=_worker, args=(video, run_id, start, req.max_frames), daemon=True)
    JOB.thread.start()
    return {"run_id": run_id}


@app.post("/api/stop")
def stop():
    JOB.stop.set()
    return {"ok": True}


@app.get("/api/status")
def status():
    return JOB.state


@app.get("/api/preview.jpg")
def preview():
    if not JOB.preview:
        return Response(status_code=204)
    return Response(JOB.preview, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ results
@app.get("/api/runs")
def runs():
    return db.list_runs(DB)


@app.get("/api/summary")
def summary(run_id: Optional[int] = None):
    rid = _run(run_id)
    if rid is None:
        return {"run": None, "totals": None}
    return {"run": db.get_run(DB, rid), "totals": db.get_totals(DB, rid)}


@app.get("/api/density")
def density(run_id: Optional[int] = None, limit: int = 500):
    rid = _run(run_id)
    return [] if rid is None else db.get_density(DB, rid, min(limit, 5000))


@app.get("/api/events")
def events(run_id: Optional[int] = None, limit: int = 200):
    rid = _run(run_id)
    return [] if rid is None else db.get_events(DB, rid, min(limit, 2000))


@app.get("/api/report/hourly")
def hourly(run_id: Optional[int] = None):
    rid = _run(run_id)
    return [] if rid is None else reports.hourly_report(DB, rid)


@app.get("/api/report/hourly.csv")
def hourly_csv(run_id: Optional[int] = None):
    rid = _run(run_id)
    rep = [] if rid is None else reports.hourly_report(DB, rid)
    return Response(reports.to_csv(rep), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="hourly_report_run{rid}.csv"'})


# dashboard (must be mounted last)
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
