## ANPR Integration
ANPR (Automatic Number Plate Recognition) runs **as a parallel intelligence layer alongside Hailo detection**. While Hailo tracks vehicles in real time, ANPR reads their licence plates — triggered only when a vehicle has dwelled long enough for a clean, front-facing read. It runs in a separate Docker container and is never allowed to slow down the main video pipeline.

### How it works

The relevant files are in [`basic_pipelines/`](https://github.com/VariPhiGen/hailo_ai_acts/tree/main/basic_pipelines) (and `configuration.json` at the repo root):

**`anpr_handler.py`** — A thin HTTP client with a built-in circuit breaker. It encodes a vehicle crop to JPEG and POSTs it to `/detect-plate/` on the ANPR Docker server. The response is a JSON object `{"text": "MH12AB1234"}` containing the plate string. If the server goes offline, calls are bypassed instantly and the breaker resets itself after a configurable blackout window — Hailo detection speed is never affected. Every successful read is persisted to a per-sensor JSON file on disk (`<sensor_id>_anpr.json`). The server URL is read from the `.env` file so no code changes are needed when deploying to a new environment.

**`detection.py` — Central ANPR control pipeline** — At startup, `_init_anpr_from_config` reads every active activity's config and registers those with `"anpr": 1`. If any activity configures `anpr_interval > 0`, a single dedicated background thread (`_anpr_thread_loop`) fires ANPR on the full frame at that interval and stores results in a shared `anpr_results` dict (protected by `anpr_lock`) — this is **Mode A (interval-based)**. Activities that set `anpr_interval: 0` get access to `self.parent.anpr_handler` directly and fire on their own terms — this is **Mode B (event-driven)**. Hailo and ANPR never block each other regardless of mode.

**`activities.py` — AnprTest activity** — A test activity demonstrating **Mode B (event-driven)**. It tracks vehicles (car, truck, bus, motorcycle) by Hailo tracker ID. Once a vehicle has been visible for longer than `dwell_limit` seconds, it starts a multi-frame sampling window: up to `anpr_max_samples` crops are sent to the ANPR server over `anpr_sample_window` seconds, spaced `anpr_sample_interval` seconds apart. Each sample is dispatched from a one-off daemon thread so the pipeline never stalls. A **majority vote** across all samples picks the final plate — a reading that only appears once is treated as noise and discarded. Vehicles with a bounding-box width-to-height ratio above `anpr_side_on_threshold` are skipped entirely because they are passing side-on and the plate is not visible.

### Two modes — controlled from `configuration.json`

| Mode | `anpr_interval` | Behaviour |
|---|---|---|
| **A — Interval** | `> 0` | Background thread runs ANPR on the full frame every N seconds |
| **B — Event-driven** | `0` | Activity fires ANPR only when a specific condition is met |

AnprTest uses **Mode B**. Use Mode A for activities that need a continuous sweep.

**Example config:**
```json
"AnprTest": {
  "parameters": {
    "subcategory_mapping": ["car", "truck", "bus", "motorcycle"],
    "anpr": 1,
    "anpr_interval": 0,
    "dwell_limit": 5,
    "anpr_sample_window": 8,
    "anpr_max_samples": 5,
    "anpr_sample_interval": 1.5,
    "anpr_min_votes": 2,
    "anpr_side_on_threshold": 3.5,
    "timezone": "Asia/Kolkata"
  }
}
```

### Environment setup

Add to your `.env` file before running:
```env
ANPR_SERVER_URL=https://your-anpr-server/detect-plate
```
The handler prints the resolved endpoint at startup so you can immediately confirm it is correct:
```
[ANPR] Handler ready.
[ANPR]   Endpoint : https://your-anpr-server/detect-plate
[ANPR]   JSON log : <sensor_id>_anpr.json
```

### Adding ANPR to your own activity (Mode B)

Add these keys to the activity's `parameters` block in `configuration.json`:
```json
"anpr": 1,
"anpr_interval": 0
```
Then inside your activity method, crop the vehicle and dispatch from a daemon thread:
```python
import threading

crop = self.parent.image[y1:y2, x1:x2].copy()
t = threading.Thread(
    target=lambda: self.parent.anpr_handler.process_frame(
        crop,
        activity_name="YourActivity",
        extra_meta={"tracker_id": tid}
    ),
    daemon=True
)
t.start()
```
The result is automatically persisted to `<sensor_id>_anpr.json` and logged to console.

---

## Face Recognition Integration

Face Recognition runs **as a third parallel intelligence layer** on top of Hailo detection. While Hailo tracks persons in real time, Face Recognition identifies who they are — querying the API only on first appearance of each new tracker ID, with configurable re-checks for unknowns. It runs in a separate Docker container and is fully non-blocking.

### How it works

The relevant files are in [`basic_pipelines/`](https://github.com/VariPhiGen/hailo_ai_acts/tree/main/basic_pipelines) (and `configuration.json` at the repo root):

**`face_recognition_handler.py`** — A batch-capable HTTP client with a built-in circuit breaker. Its key difference from the ANPR handler is that it sends **multiple face crops in a single POST request** — all persons visible in the current tick are packaged into one multipart request to `/image_match/`. Each crop is named `tid_<tracker_id>.jpg` so results can be mapped back to the correct Hailo tracker. The API responds with a JSON array:
```json
{"results": [
    {"image_name": "tid_42.jpg", "status": "match_found", "person_name": "Alice", "confidence": 97.8},
    {"image_name": "tid_7.jpg",  "status": "no_match"}
]}
```
Every match is persisted to `<sensor_id>_facerec.json`. The `match_face()` convenience method wraps the batch call for single-image use. The server URL is read from `.env`.

**`detection.py` — Central Face Recognition control pipeline** — At startup, `_init_facerec_from_config` reads every active activity's config and registers those with `"face_rec": 1`. A dedicated background thread (`_facerec_thread_loop`) handles **Mode A (interval-based)**: every N seconds it finds all visible person detections, crops the face region (top 35% of each person bounding box), filters out crops that are too small or side-facing, then batch-sends everything in one API call. Results are stored in a shared `facerec_results` dict keyed by tracker ID (protected by `facerec_lock`). Activities read from this dict using `self.parent.get_facerec_result(tracker_id)`. For **Mode B (event-driven)**, activities call `self.parent.facerec_handler.match_faces()` directly from their own daemon thread.

**`activities.py` — FaceRecognition activity** — A test activity demonstrating **Mode B (event-driven)**. It fires once per new tracker ID the moment a person first appears, collecting all new persons in the current tick into a single batch request rather than one request per person. Two quality guards are applied before any crop is sent: a **size guard** (skip if the face region is smaller than `facerec_min_face_px` — person is too far away) and a **side-on guard** (skip if face width-to-height ratio is below `facerec_side_on_threshold` — person is in profile). Confirmed identities are stored permanently for the session so the same person is never re-queried. Unrecognised persons can be re-queried after `facerec_recheck_interval` seconds.

### Two modes — controlled from `configuration.json`

| Mode | `facerec_interval` | Behaviour |
|---|---|---|
| **A — Interval** | `> 0` | Background thread batches all visible persons and queries every N seconds |
| **B — Event-driven** | `0` | Activity fires only on first appearance of each new tracker ID |

FaceRecognition uses **Mode B**. Use Mode A for access-control scenarios where you want continuous sweeps regardless of whether a tracker is new.

**Example config:**
```json
"FaceRecognition": {
  "parameters": {
    "subcategory_mapping": ["person"],
    "face_rec": 1,
    "facerec_interval": 0,
    "facerec_min_face_px": 50,
    "facerec_side_on_threshold": 0.4,
    "facerec_recheck_interval": 60,
    "timezone": "Asia/Kolkata"
  }
}
```

### Environment setup

Add to your `.env` file before running:
```env
FACEREC_SERVER_URL=http://your-facerec-server/image_match/
```
The handler prints the resolved endpoint at startup:
```
[FaceRec] Handler ready.
[FaceRec]   Endpoint : http://your-facerec-server/image_match/
[FaceRec]   JSON log : <sensor_id>_facerec.json
```

### Adding Face Recognition to your own activity (Mode B)

Add to the activity's `parameters` block in `configuration.json`:
```json
"face_rec": 1,
"facerec_interval": 0
```
Then inside your activity method, read an already-identified result for free (no API call):
```python
result = self.parent.get_facerec_result(tracker_id)
if result:
    print(result["person_name"], result["confidence"])
```
Or dispatch a fresh batch query from a daemon thread when you need it:
```python
import threading

crops = [
    {"tracker_id": tid, "crop": face_crop.copy(), "extra_meta": {"zone": zone_name}}
    for tid, face_crop in pending_crops.items()
]
t = threading.Thread(
    target=lambda: self.parent.facerec_handler.match_faces(
        crops,
        activity_name="YourActivity"
    ),
    daemon=True
)
t.start()
```
Results are automatically persisted to `<sensor_id>_facerec.json` and the `facerec_results` dict is updated so other activities can read them immediately after the thread completes.

To test ANPR and Face Recognition, follow the Quick Start steps below.

## Quick Start

Clone the repo and work from its root directory:
```
git clone https://github.com/VariPhiGen/hailo_ai_acts.git && cd hailo_ai_acts
```

Run these commands in order from the project root (all commands assume you are inside `hailo_ai_acts`):

1) Base system setup  
```
bash setup_system.sh
sudo reboot
```
Prep base packages/drivers; reboot to apply.

2) Install project dependencies  
```
bash install.sh
```
Installs Python deps and local tools.

3) Download required resources (models, etc.)  
```
bash download_hailo8l.sh
```
Fetches required models/resources.

4) Smoke test the simple detector  
```
source setup_env.sh
python basic_pipelines/detection_simple.py
```
Verifies environment and minimal pipeline run.

If you hit common errors, run the helper:
```
bash Common_Error.sh
```
Attempts common fixes; rerun your test after this if needed.

5) AI environment setup  
```
bash setup_ota_env.sh
bash setup_ai_env.sh
```
Sets up additional AI environment pieces.

6) Write the default configuration file  
```
bash write_configuration.sh
```
Generates `configuration.json` with default sensor/camera/activity settings.

7) Full detection test with display (verify broker/S3/detections)  
```
bash run_detection_with_display.sh
```
Make sure the Kafka broker is connected, S3 uploads succeed, and detections look correct **before** proceeding.

8) Optional: install as a service (only after verification above)  
```
bash setup_service.sh
```
Registers the pipeline as a service once confirmed working interactively.



If any script needs a custom path or args, pass them as needed (defaults assume project root). For `.env`, ensure it exists before sourcing `setup_env.sh`.  
#Variphi Acts on Hailo
