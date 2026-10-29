import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, reports  # noqa: E402


def test_db_and_hourly_report(tmp_path):
    p = tmp_path / "t.db"
    db.init_db(p)
    start = datetime(2026, 10, 29, 8, 50, 0)
    rid = db.create_run(p, "x.mp4", start)
    crossings = [
        {"video_time": 10, "track_id": 1, "line": "L", "direction": "down", "vclass": "car"},
        {"video_time": 20, "track_id": 2, "line": "L", "direction": "up", "vclass": "bus"},
        {"video_time": 700, "track_id": 3, "line": "L", "direction": "down", "vclass": "truck"},  # 09:01
    ]
    density = [
        {"video_time": 10, "vehicles": 4.0, "peak": 6, "occupancy": 0.2, "avg_speed": 0.1, "level": "moderate"},
        {"video_time": 700, "vehicles": 9.0, "peak": 11, "occupancy": 0.5, "avg_speed": 0.02, "level": "congested"},
    ]
    events = [{"video_time": 705, "track_id": 3, "type": "wrong_way", "vclass": "truck", "details": "d"}]
    db.insert_batch(p, rid, start, crossings, density, events)
    rep = reports.hourly_report(p, rid)
    assert [r["hour"] for r in rep] == ["2026-10-29 08:00", "2026-10-29 09:00"]
    assert rep[0]["total_vehicles"] == 2 and rep[0]["by_class"]["bus"] == 1
    assert rep[1]["congested_pct"] == 100.0 and rep[1]["wrong_way_events"] == 1
    assert "hour,total_vehicles" in reports.to_csv(rep)
    t = db.get_totals(p, rid)
    assert t["total_vehicles"] == 3 and t["wrong_way_events"] == 1


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_CONFIG", str(tmp_path / "c.yaml"))
    import yaml
    from app.config import ROOT
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["paths"] = {"database": str(tmp_path / "api.db"), "videos": str(tmp_path / "vids")}
    yaml.safe_dump(cfg, open(tmp_path / "c.yaml", "w"))
    for m in [m for m in sys.modules if m.startswith("app.main")]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    from app import main
    c = TestClient(main.app)
    assert c.get("/api/summary").json()["run"] is None
    assert c.get("/api/report/hourly").json() == []
    rid = db.create_run(main.DB, "demo", datetime(2026, 10, 29, 8, 0, 0))
    db.insert_batch(main.DB, rid, datetime(2026, 10, 29, 8, 0, 0),
                    [{"video_time": 1, "track_id": 1, "line": "L", "direction": "down", "vclass": "car"}])
    assert c.get("/api/summary").json()["totals"]["total_vehicles"] == 1
    assert c.get("/api/report/hourly.csv").status_code == 200
    assert c.post("/api/process", json={"video": "nope.mp4"}).status_code == 404
    assert c.get("/api/status").json()["status"] == "idle"
    assert c.get("/").status_code in (200, 404)
