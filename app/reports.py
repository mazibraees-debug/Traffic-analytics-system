"""Hourly traffic reports built from the SQLite tables."""
from __future__ import annotations

import csv
import io

from . import db

CLASSES = ["car", "motorcycle", "bus", "truck"]
HOUR = "strftime('%Y-%m-%d %H:00', ts)"


def hourly_report(path, run_id: int) -> list[dict]:
    rows: dict[str, dict] = {}

    def row(h):
        return rows.setdefault(h, {
            "hour": h, "total_vehicles": 0,
            "by_class": {c: 0 for c in CLASSES}, "by_line": {},
            "avg_vehicles": None, "peak_vehicles": None, "avg_occupancy_pct": None,
            "avg_speed": None, "congested_pct": 0.0, "heavy_pct": 0.0,
            "wrong_way_events": 0})

    with db.session(path) as c:
        for r in c.execute(
                f"SELECT {HOUR} h, vclass, line, direction, COUNT(*) n FROM crossings "
                "WHERE run_id=? GROUP BY 1,2,3,4", (run_id,)):
            x = row(r["h"])
            x["total_vehicles"] += r["n"]
            x["by_class"][r["vclass"]] = x["by_class"].get(r["vclass"], 0) + r["n"]
            key = f"{r['line']} ({r['direction']})"
            x["by_line"][key] = x["by_line"].get(key, 0) + r["n"]
        for r in c.execute(
                f"SELECT {HOUR} h, AVG(vehicles) av, MAX(peak) pk, AVG(occupancy) oc, AVG(avg_speed) sp, "
                "COUNT(*) n, SUM(level='congested') cg, SUM(level='heavy') hv FROM density "
                "WHERE run_id=? GROUP BY 1", (run_id,)):
            x = row(r["h"])
            x["avg_vehicles"] = round(r["av"], 2)
            x["peak_vehicles"] = r["pk"]
            x["avg_occupancy_pct"] = round(100 * r["oc"], 1)
            x["avg_speed"] = None if r["sp"] is None else round(r["sp"], 3)
            x["congested_pct"] = round(100.0 * (r["cg"] or 0) / r["n"], 1)
            x["heavy_pct"] = round(100.0 * (r["hv"] or 0) / r["n"], 1)
        for r in c.execute(
                f"SELECT {HOUR} h, COUNT(*) n FROM events WHERE run_id=? AND type='wrong_way' GROUP BY 1",
                (run_id,)):
            row(r["h"])["wrong_way_events"] = r["n"]
    return [rows[k] for k in sorted(rows)]


def to_csv(report: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["hour", "total_vehicles", *CLASSES, "avg_vehicles_in_roi", "peak_vehicles_in_roi",
                "avg_occupancy_pct", "avg_speed", "heavy_pct", "congested_pct", "wrong_way_events"])
    for r in report:
        w.writerow([r["hour"], r["total_vehicles"], *[r["by_class"].get(c, 0) for c in CLASSES],
                    r["avg_vehicles"], r["peak_vehicles"], r["avg_occupancy_pct"], r["avg_speed"],
                    r["heavy_pct"], r["congested_pct"], r["wrong_way_events"]])
    return buf.getvalue()
