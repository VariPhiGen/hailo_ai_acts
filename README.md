## YOLOE Integration

YOLOE is an open-vocabulary vision model that runs **in parallel with Hailo detection** as a second intelligence layer. While Hailo handles fast real-time detection, YOLOE can find things Hailo's fixed model was not trained on — using either text prompts, a visual reference image, or both.

### How it works

The relevant files are in [`basic_pipelines/`](https://github.com/VariPhiGen/hailo_ai_acts/tree/yoloe/basic_pipelines) (and `configuration.json` at the repo root):

**`yoloe_handler.py`** — A unified HTTP client with a built-in circuit breaker. It encodes the current frame and routes it to one of three endpoints on the YOLOE Docker server — `/predict_prompt` (text mode), `/predict_visual` (visual mode), or /predict_both — based on the activity's configured mode. If the server goes offline, it instantly bypasses YOLOE and pings every 5 seconds to check if it's back, so Hailo detection speed is never affected. Results are persisted to a per-sensor JSON file on disk and also returned to the central thread in `detection.py.`

**`detection.py` — Central YOLOE control pipeline** — At startup, `_init_yoloe_from_config` reads every active activity's config and registers those with `"yoloe": 1`. A single dedicated background thread (`_yoloe_thread_loop`) then runs continuously, calling the YOLOE server for each registered activity at its configured interval and storing results in a shared `yoloe_results` dict (protected by `yoloe_lock`). Activities read from this dict independently — YOLOE and Hailo never block each other.

**`activities.py` — YoloeTest activity** — A test activity that every second prints a side-by-side summary of what Hailo detected (persons by tracker ID) and what YOLOE detected (text-based and visual-based detections separately, filtered by confidence and `condition_label`).

### Three modes — all controlled from `configuration.json`

Under `activities_data → YoloeTest → parameters`, change `yoloe_mode` to switch modes:

| `yoloe_mode` | What YOLOE does |
|---|---|
| `"text"` | Detects objects matching your `condition_label` text prompts (e.g. `"scaffolding"`) |
| `"visual"` | Detects objects matching a reference image — annotate a reference image via the YOLOE app UI (see **[VariPhiGen/YOLOe](https://github.com/VariPhiGen/YOLOe)** for setup and annotation steps) before testing this mode |
| `"both"` | Runs both; results are tagged `"source": "text"` or `"source": "visual"` |

No code changes are needed — only the config changes.

**Example config for text-only mode:**
```json
"YoloeTest": {
  "parameters": {
    "subcategory_mapping": ["person"],
    "yoloe": 1,
    "yoloe_mode": "text",
    "condition_label": ["scaffolding"],
    "yoloe_confidence": 0.02,
    "yoloe_interval": 5
  }
}
```
Change `"yoloe_mode"` to `"visual"` or `"both"` and update `condition_label` with your target object names to test the other modes.

### Adding YOLOE to your own activity

To add YOLOE to any existing activity, add these keys to its `parameters` block in `configuration.json`:
```json
"yoloe": 1,
"yoloe_mode": "text",
"condition_label": ["your_object"],
"yoloe_confidence": 0.1,
"yoloe_interval": 5
```
Then in `activities.py`, read the result inside your activity method using:
```python
with self.parent.yoloe_lock:
    activity_data = self.parent.yoloe_results.get("YourActivityName")
    if activity_data:
        result = activity_data["result"]
```

Before testing, set up the YOLOE server (handles text prompts, visual prompts, and the visual prompter UI) by following the guidelines in **[VariPhiGen/YOLOe](https://github.com/VariPhiGen/YOLOe)**. Then follow the Quick Start steps below.

To test YOLOE, follow the Quick Start steps below.

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