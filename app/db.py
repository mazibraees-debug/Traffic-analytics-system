"""SQLite storage. One short-lived connection per call keeps it thread-safe."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

FMT = "%Y-%m-%d %H:%M:%S"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    start_time TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    frames INTEGER DEFAULT 0,
    duration_s REAL DEFAULT 0,
    message TEXT
);
CREATE TABLE IF NOT EXISTS crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts TEXT NOT NULL, video_time REAL, track_id INTEGER,
    line TEXT NOT NULL, direction TEXT NOT NULL, vclass TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS density (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts TEXT NOT NULL, video_time REAL,
    vehicles REAL, peak INTEGER, occupancy REAL, avg_speed REAL, level TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts TEXT NOT NULL, video_time REAL, track_id INTEGER,
    type TEXT NOT NULL, vclass TEXT, details TEXT
);
CREATE INDEX IF NOT EXISTS ix_cross_run_ts ON crossings(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_dens_run_ts ON density(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_evt_run_ts ON events(run_id, ts);
"""


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session(path):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with session(path) as c:
        c.executescript(SCHEMA)


def _ts(start: datetime, video_time: float) -> str:
    return (start + timedelta(seconds=video_time or 0)).strftime(FMT)


def create_run(path, source: str, start_time: datetime) -> int:
    with session(path) as c:
        cur = c.execute(
            "INSERT INTO runs(source,start_time,created_at,status) VALUES (?,?,?,'running')",
            (source, start_time.strftime(FMT), datetime.now().strftime(FMT)))
        return cur.lastrowid


def finish_run(path, run_id: int, status: str, frames: int = 0,
               duration_s: float = 0.0, message: Optional[str] = None):
    with session(path) as c:
        c.execute("UPDATE runs SET status=?, frames=?, duration_s=?, message=? WHERE id=?",
                  (status, frames, duration_s, message, run_id))


def insert_batch(path, run_id: int, start: datetime, crossings=(), density=(), events=()):
    if not (crossings or density or events):
        return
    with session(path) as c:
        c.executemany(
            "INSERT INTO crossings(run_id,ts,video_time,track_id,line,direction,vclass) VALUES (?,?,?,?,?,?,?)",
            [(run_id, _ts(start, x["video_time"]), x["video_time"], x["track_id"],
              x["line"], x["direction"], x["vclass"]) for x in crossings])
        c.executemany(
            "INSERT INTO density(run_id,ts,video_time,vehicles,peak,occupancy,avg_speed,level) VALUES (?,?,?,?,?,?,?,?)",
            [(run_id, _ts(start, x["video_time"]), x["video_time"], x["vehicles"], x["peak"],
              x["occupancy"], x["avg_speed"], x["level"]) for x in density])
        c.executemany(
            "INSERT INTO events(run_id,ts,video_time,track_id,type,vclass,details) VALUES (?,?,?,?,?,?,?)",
            [(run_id, _ts(start, x["video_time"]), x["video_time"], x["track_id"],
              x["type"], x.get("vclass"), x.get("details")) for x in events])


# ---------------------------------------------------------------- queries
def list_runs(path, limit=50):
    with session(path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]


def get_run(path, run_id: int):
    with session(path) as c:
        r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(r) if r else None


def latest_run_id(path) -> Optional[int]:
    with session(path) as c:
        r = c.execute("SELECT MAX(id) AS m FROM runs").fetchone()
        return r["m"]


def get_density(path, run_id: int, limit=500):
    with session(path) as c:
        rows = c.execute(
            "SELECT * FROM (SELECT * FROM density WHERE run_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
            (run_id, limit))
        return [dict(r) for r in rows]


def get_events(path, run_id: int, limit=200):
    with session(path) as c:
        rows = c.execute("SELECT * FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit))
        return [dict(r) for r in rows]


def get_totals(path, run_id: int):
    with session(path) as c:
        by_class = {r["vclass"]: r["n"] for r in c.execute(
            "SELECT vclass, COUNT(*) n FROM crossings WHERE run_id=? GROUP BY vclass", (run_id,))}
        by_line = [dict(r) for r in c.execute(
            "SELECT line, direction, COUNT(*) n FROM crossings WHERE run_id=? GROUP BY line, direction ORDER BY line, direction",
            (run_id,))]
        d = c.execute(
            "SELECT AVG(vehicles) avg_v, MAX(peak) peak_v, AVG(avg_speed) spd, COUNT(*) n, "
            "SUM(level='congested') cong, SUM(level='heavy') heavy FROM density WHERE run_id=?",
            (run_id,)).fetchone()
        last = c.execute("SELECT * FROM density WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
        ww = c.execute("SELECT COUNT(*) n FROM events WHERE run_id=? AND type='wrong_way'", (run_id,)).fetchone()["n"]
    n = d["n"] or 0
    return {
        "total_vehicles": sum(by_class.values()),
        "by_class": by_class,
        "by_line": by_line,
        "avg_vehicles": d["avg_v"],
        "peak_vehicles": d["peak_v"],
        "avg_speed": d["spd"],
        "congested_pct": round(100.0 * (d["cong"] or 0) / n, 1) if n else 0.0,
        "heavy_pct": round(100.0 * (d["heavy"] or 0) / n, 1) if n else 0.0,
        "latest_level": last["level"] if last else None,
        "latest_vehicles": last["vehicles"] if last else None,
        "wrong_way_events": ww,
    }
