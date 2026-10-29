"""YOLO + ByteTrack wrapper (via Ultralytics). Imported lazily so the rest of
the project (analytics, API, reports) works without torch installed."""
from __future__ import annotations

from .analytics import Track
from .config import ROOT, resolve


class YoloByteTrackDetector:
    def __init__(self, det_cfg: dict):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install -r requirements.txt") from e
        self.cfg = det_cfg
        model_name = det_cfg.get("model", "yolov8n.pt")
        local = ROOT / "models" / model_name
        self.model = YOLO(str(local) if local.exists() else model_name)
        self.names = {int(k): v for k, v in det_cfg["classes"].items()}
        self.tracker = str(resolve(det_cfg.get("tracker", "configs/bytetrack_cpu.yaml")))

    def track(self, frame) -> list[Track]:
        h, w = frame.shape[:2]
        res = self.model.track(
            frame, persist=True, tracker=self.tracker, verbose=False,
            conf=self.cfg.get("conf", 0.3), iou=self.cfg.get("iou", 0.5),
            imgsz=self.cfg.get("imgsz", 480), classes=list(self.names),
            device=self.cfg.get("device", "cpu"))[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        out = []
        for (x1, y1, x2, y2), i, c in zip(xyxy, ids, cls):
            out.append(Track(int(i), self.names.get(int(c), str(c)),
                             (float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h)))
        return out


def load_detector(det_cfg: dict):
    return YoloByteTrackDetector(det_cfg)
