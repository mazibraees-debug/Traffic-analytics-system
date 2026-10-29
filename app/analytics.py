"""Pure-python traffic analytics: line counting, density, congestion, wrong-way.

Nothing here depends on OpenCV or YOLO, so it is fast to unit test and can be
fed by any detector/tracker that produces `Track` objects. All coordinates are
normalised to 0..1.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

Point = tuple[float, float]
LEVELS = ["free", "moderate", "heavy", "congested"]


@dataclass
class Track:
    id: int
    cls: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (normalised)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def ref_point(bbox, mode: str = "bottom_center") -> Point:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    return (cx, y2) if mode == "bottom_center" else (cx, (y1 + y2) / 2.0)


def side(a: Point, b: Point, c: Point) -> float:
    """>0 if c is on the 'positive' side of the directed line a->b."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 crosses segment q1-q2 (zero treated as positive
    so a vehicle that lands exactly on the line is still counted once)."""
    s1, s2 = side(q1, q2, p1) >= 0, side(q1, q2, p2) >= 0
    s3, s4 = side(p1, p2, q1) >= 0, side(p1, p2, q2) >= 0
    return s1 != s2 and s3 != s4


def point_in_polygon(pt: Point, poly) -> bool:
    if not poly:
        return True
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def polygon_area(poly) -> float:
    if not poly:
        return 1.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


# --------------------------------------------------------------------------
# motion history shared by speed + wrong-way
# --------------------------------------------------------------------------
class Motion:
    def __init__(self, maxlen: int = 30):
        self.maxlen = maxlen
        self.hist: dict[int, deque] = {}
        self.last_frame: dict[int, int] = {}
        self.n = 0

    def update(self, tracks, video_time: float, ref_mode: str):
        self.n += 1
        for t in tracks:
            p = ref_point(t.bbox, ref_mode)
            d = self.hist.setdefault(t.id, deque(maxlen=self.maxlen))
            d.append((video_time, p[0], p[1]))
            self.last_frame[t.id] = self.n
        if self.n % 300 == 0:
            stale = [i for i, f in self.last_frame.items() if self.n - f > 300]
            for i in stale:
                self.hist.pop(i, None)
                self.last_frame.pop(i, None)

    def get(self, tid: int):
        return self.hist.get(tid, ())

    def speed(self, tid: int, window: int = 6) -> Optional[float]:
        """Frame-widths per second over the last `window` samples."""
        h = list(self.hist.get(tid, ()))[-window:]
        if len(h) < 3:
            return None
        dt = h[-1][0] - h[0][0]
        if dt <= 0:
            return None
        return math.hypot(h[-1][1] - h[0][1], h[-1][2] - h[0][2]) / dt


# --------------------------------------------------------------------------
# virtual line counter
# --------------------------------------------------------------------------
class LineCounter:
    def __init__(self, lines, ref_mode="bottom_center"):
        self.lines = lines or []
        self.ref = ref_mode
        self.prev: dict[int, Point] = {}
        self.seen: dict[int, int] = {}
        self.counted: set = set()
        self.n = 0

    def update(self, tracks, video_time: float) -> list[dict]:
        out = []
        self.n += 1
        for t in tracks:
            p = ref_point(t.bbox, self.ref)
            prev = self.prev.get(t.id)
            if prev is not None:
                for ln in self.lines:
                    key = (t.id, ln["name"])
                    if key in self.counted:
                        continue
                    a, b = tuple(ln["p1"]), tuple(ln["p2"])
                    if segments_cross(prev, p, a, b):
                        positive = side(a, b, p) >= 0
                        label = (ln.get("positive_label", "positive") if positive
                                 else ln.get("negative_label", "negative"))
                        self.counted.add(key)
                        out.append({"video_time": video_time, "track_id": t.id,
                                    "line": ln["name"], "direction": label,
                                    "vclass": t.cls})
            self.prev[t.id] = p
            self.seen[t.id] = self.n
        if self.n % 300 == 0:
            stale = {i for i, f in self.seen.items() if self.n - f > 300}
            for i in stale:
                self.prev.pop(i, None)
                self.seen.pop(i, None)
            self.counted = {k for k in self.counted if k[0] not in stale}
        return out


# --------------------------------------------------------------------------
# wrong-way detection
# --------------------------------------------------------------------------
class WrongWayDetector:
    def __init__(self, cfg: dict):
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", True)
        self.history = int(cfg.get("history", 12))
        self.min_disp = float(cfg.get("min_displacement", 0.05))
        self.confirm = int(cfg.get("confirm_frames", 4))
        self.cos_thr = math.cos(math.radians(cfg.get("angle_threshold_deg", 120)))
        self.zones = []
        for z in cfg.get("zones", []) or []:
            ex, ey = z["expected_direction"]
            norm = math.hypot(ex, ey) or 1.0
            self.zones.append({"name": z.get("name", "zone"),
                               "polygon": z.get("polygon"),
                               "dir": (ex / norm, ey / norm)})
        self.bad: dict[int, int] = {}
        self.flagged: set[int] = set()

    def _zone_for(self, p: Point):
        for z in self.zones:
            if point_in_polygon(p, z["polygon"]):
                return z
        return None

    def update(self, tracks, motion: Motion, video_time: float) -> list[dict]:
        if not self.enabled or not self.zones:
            return []
        events = []
        for t in tracks:
            if t.id in self.flagged:
                continue
            h = list(motion.get(t.id))
            if len(h) < self.history:
                continue
            h = h[-self.history:]
            dx, dy = h[-1][1] - h[0][1], h[-1][2] - h[0][2]
            mag = math.hypot(dx, dy)
            zone = self._zone_for((h[-1][1], h[-1][2]))
            if zone is None or mag < self.min_disp:
                self.bad[t.id] = 0
                continue
            cos = (dx * zone["dir"][0] + dy * zone["dir"][1]) / mag
            self.bad[t.id] = self.bad.get(t.id, 0) + 1 if cos < self.cos_thr else 0
            if self.bad[t.id] >= self.confirm:
                self.flagged.add(t.id)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                events.append({
                    "video_time": video_time, "track_id": t.id,
                    "type": "wrong_way", "vclass": t.cls,
                    "details": f"{zone['name']}: moving {angle:.0f} deg off expected direction"})
        return events


# --------------------------------------------------------------------------
# density + congestion
# --------------------------------------------------------------------------
def classify_level(count: float, speed: Optional[float], roi_cfg: dict):
    capacity = max(1.0, float(roi_cfg.get("capacity", 12)))
    ratio = count / capacity
    if count < roi_cfg.get("min_vehicles", 3):
        return "free", ratio
    slow = speed is not None and speed < roi_cfg.get("slow_speed", 0.06)
    if ratio >= roi_cfg.get("congested_ratio", 0.70) and slow:
        return "congested", ratio
    if ratio >= roi_cfg.get("heavy_ratio", 0.50) or (slow and ratio >= roi_cfg.get("moderate_ratio", 0.25)):
        return "heavy", ratio
    if ratio >= roi_cfg.get("moderate_ratio", 0.25):
        return "moderate", ratio
    return "free", ratio


class FrameAnalyzer:
    """Feed it the tracks of each processed frame; it returns new crossings,
    events, and (every `sample_interval_s`) a density sample."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ref_mode = cfg.get("reference_point", "bottom_center")
        self.interval = float(cfg.get("sample_interval_s", 5))
        self.roi_cfg = cfg.get("roi", {}) or {}
        self.roi_poly = self.roi_cfg.get("polygon")
        self.roi_area = polygon_area(self.roi_poly)
        self.motion = Motion(maxlen=max(30, int(cfg.get("wrong_way", {}).get("history", 12)) + 2))
        self.counter = LineCounter(cfg.get("lines", []), self.ref_mode)
        self.wrong_way = WrongWayDetector(cfg.get("wrong_way", {}))
        self._recent: deque = deque(maxlen=10)
        self._reset_acc(None)
        self.live = {"count": 0, "avg_speed": None, "level": "free"}

    # accumulation window --------------------------------------------------
    def _reset_acc(self, start):
        self._start = start
        self._frames = 0
        self._sum_count = 0.0
        self._peak = 0
        self._sum_occ = 0.0
        self._sum_speed = 0.0
        self._speed_n = 0

    def _emit(self) -> Optional[dict]:
        if not self._frames:
            return None
        mean_count = self._sum_count / self._frames
        speed = self._sum_speed / self._speed_n if self._speed_n else None
        level, ratio = classify_level(mean_count, speed, self.roi_cfg)
        return {"video_time": self._start, "vehicles": round(mean_count, 2),
                "peak": self._peak, "occupancy": round(self._sum_occ / self._frames, 4),
                "avg_speed": None if speed is None else round(speed, 4),
                "level": level}

    # main entry -----------------------------------------------------------
    def update(self, tracks: list[Track], video_time: float) -> dict:
        self.motion.update(tracks, video_time, self.ref_mode)
        crossings = self.counter.update(tracks, video_time)
        events = self.wrong_way.update(tracks, self.motion, video_time)

        in_roi = [t for t in tracks
                  if point_in_polygon(ref_point(t.bbox, self.ref_mode), self.roi_poly)]
        count = len(in_roi)
        occ = min(1.0, sum(bbox_area(t.bbox) for t in in_roi) / self.roi_area)
        speeds = [s for s in (self.motion.speed(t.id) for t in in_roi) if s is not None]
        avg_speed = sum(speeds) / len(speeds) if speeds else None

        # per-frame smoothed state for the live overlay
        self._recent.append((count, avg_speed))
        rc = sum(c for c, _ in self._recent) / len(self._recent)
        rs = [s for _, s in self._recent if s is not None]
        rspeed = sum(rs) / len(rs) if rs else None
        level, _ = classify_level(rc, rspeed, self.roi_cfg)
        self.live = {"count": count, "avg_speed": rspeed, "level": level}

        # interval aggregation
        density = []
        if self._start is None:
            self._reset_acc(video_time)
        elif video_time - self._start >= self.interval:
            s = self._emit()
            if s:
                density.append(s)
            self._reset_acc(video_time)
        self._frames += 1
        self._sum_count += count
        self._peak = max(self._peak, count)
        self._sum_occ += occ
        if avg_speed is not None:
            self._sum_speed += avg_speed
            self._speed_n += 1
        return {"crossings": crossings, "events": events, "density": density}

    def finish(self) -> list[dict]:
        s = self._emit()
        self._reset_acc(None)
        return [s] if s else []
