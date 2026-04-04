## Pose Estimation

Pose estimation runs **in parallel with object detection** inside [`basic_pipelines/detection.py`](https://github.com/VariPhiGen/hailo_ai_acts/tree/pose_estimation/basic_pipelines).

To enable it, simply set the `pose_hef_path` key in your `configuration.json` pointing to your pose HEF model file — no code changes needed. If the path is valid, pose estimation activates automatically. If it is missing or invalid, the pipeline runs in detection-only mode.

The following pose data is then available per frame via `user_data`:
- `pose_keypoints` — keypoint lists per detected person
- `pose_tracker_ids` — tracker IDs for persons with pose
- `pose_detection_boxes` — bounding boxes for those persons
- `pose_classes` — class labels (all `"person"`)

For a detailed understanding of how pose estimation is implemented end-to-end, refer to:
📄 **[Pose Estimation Implementation Guide](https://github.com/VariPhiGen/hailo-apps-infra/tree/pose_estimation)**

To test pose estimation, follow the Quick Start steps below.

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

6) Full detection test with display (verify broker/S3/detections)  
```
bash run_detection_with_display.sh
```
Make sure the Kafka broker is connected, S3 uploads succeed, and detections look correct **before** proceeding.

7) Optional: install as a service (only after verification above)  
```
bash setup_service.sh
```
Registers the pipeline as a service once confirmed working interactively.



If any script needs a custom path or args, pass them as needed (defaults assume project root). For `.env`, ensure it exists before sourcing `setup_env.sh`.  
#Variphi Acts on Hailo
