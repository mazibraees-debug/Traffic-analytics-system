# Intelligent Traffic Analytics System

Detect and track cars, buses, motorcycles and trucks in recorded traffic video, then:

- **Count vehicles** crossing virtual lines (per direction, per vehicle type)
- **Estimate density** and classify congestion (`free` / `moderate` / `heavy` / `congested`)
- **Track movement** (speed estimate per vehicle) and **flag wrong-way driving**
- **Generate hourly reports** (JSON, CSV, and a table on the dashboard)

**Stack:** YOLO (Ultralytics, `yolov8n`) · ByteTrack · OpenCV · FastAPI · SQLite · Chart.js dashboard
**Detection defaults:** frames are processed up to 1920 px wide, inference uses 1280 px, and low-confidence detections are retained to help find distant/small vehicles. Frames are sampled at ~8 fps. These settings improve recall at the cost of CPU speed; all are tunable in `config.yaml`.

```
video ──► YOLOv8n ──► ByteTrack ──► FrameAnalyzer ──► SQLite ──► FastAPI ──► dashboard
          (CPU, 1280px) (IDs)       line counts          crossings   /api/*      live preview,
                                    density/congestion   density     CSV         charts, reports
                                    wrong-way            events
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# optional, smaller CPU-only PyTorch:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn app.main:app --reload          # open http://localhost:8000
```

In the dashboard: **Upload video** → **Start analysis**. The live view shows boxes, track IDs, the counting line and the traffic zone. The first run downloads `yolov8n.pt` (~6 MB) automatically (or place it in `models/`).

### Try it without a video or model
```bash
python scripts/seed_demo.py --hours 3      # simulates a morning rush hour incl. an incident
uvicorn app.main:app                       # dashboard now shows 3 hourly rows
```
The simulation drives the real analytics engine, so it demonstrates line counting, congestion and wrong-way events.

### Command line
```bash
python -m app.cli data/videos/clip.mp4 --start-time 2026-10-29T08:00:00
```
`--start-time` is the wall-clock time of the first frame; it is what makes the **hourly** report meaningful for recorded footage.

## Calibrate for your camera (important)
Everything is in `config.yaml` using **normalised coordinates** (0–1), so it is independent of resolution.

```bash
python scripts/preview_config.py data/videos/clip.mp4     # writes data/calibration.jpg
```
It draws the counting lines, density zone and wrong-way zones (with arrows) on a frame. Edit `config.yaml`, re-run, repeat.

| Setting | What it does |
|---|---|
| `analytics.lines` | Virtual lines; a vehicle is counted once per line when its road-contact point crosses it. Add as many as you like. `positive_label`/`negative_label` name the two crossing directions. |
| `analytics.roi` | Density zone. `capacity` = vehicles that means "full"; level thresholds are ratios of capacity; `slow_speed` separates "flowing" from "crawling". |
| `analytics.wrong_way.zones` | Each zone has an `expected_direction` vector (`[0,1]` = down the screen). A vehicle moving > `angle_threshold_deg` against it, for `confirm_frames` in a row, is flagged once. Default assumes a two-way road split down the middle – **change it to match your scene**. |
| `video.target_fps`, `video.process_width`, `detector.imgsz`, `detector.conf` | Speed/recall trade-off on CPU. Higher resolutions and lower confidence thresholds find more small/distant vehicles but take longer and may add false detections. Smaller input videos are never enlarged. |

**Congestion logic:** over each 5 s window, `level` combines how full the zone is (vehicles / capacity) with average speed. Dense *and* slow = `congested`; dense but moving = `heavy`.

## API
| Endpoint | Purpose |
|---|---|
| `POST /api/upload` · `GET /api/videos` | Manage videos in `data/videos` |
| `POST /api/process` `{video, start_time?, max_frames?}` · `POST /api/stop` · `GET /api/status` | Run / stop / poll an analysis |
| `GET /api/preview.jpg` | Latest annotated frame |
| `GET /api/runs` · `GET /api/summary` | Runs and totals |
| `GET /api/density` · `GET /api/events` | Time series and wrong-way events |
| `GET /api/report/hourly` · `GET /api/report/hourly.csv` | Hourly report |

All result endpoints take `?run_id=` (default: latest). Interactive docs: `http://localhost:8000/docs`.

## Layout
```
app/analytics.py   line counter, density/congestion, wrong-way (pure python, unit-tested)
app/detector.py    YOLO + ByteTrack wrapper (lazy import)
app/pipeline.py    video loop, frame skipping, overlay, DB writes
app/db.py          SQLite schema + queries      app/reports.py  hourly report + CSV
app/main.py        FastAPI                      app/static/     dashboard
configs/bytetrack_cpu.yaml   ByteTrack settings tuned for low FPS
scripts/           seed_demo.py, preview_config.py
tests/             pytest suite (analytics, DB/API, pipeline with a fake detector)
```
Run tests: `python -m pytest -q tests`

## Notes & limits
- Speeds are in *frame-widths per second* (camera perspective is not corrected), good for relative congestion, not km/h.
- Best with a fixed camera at an elevated angle. Strong occlusion or very small distant vehicles can still reduce accuracy with `yolov8n`; try `yolov8s.pt` if you have CPU headroom. No detector can guarantee every vehicle is found.
- Tracker ID switches can occasionally double-count a vehicle; each (track, line) pair is only counted once.
