#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-configuration.json}"

# The sensor ID this script intends to write. 
# You can change this manually before running, or pass it via ENV.
NEW_SENSOR_ID="SENSOR_ID_PLACEHOLDER"

# Determine final sensor_id:
# 1. If NEW_SENSOR_ID is a real ID, use it (overwrite).
# 2. If NEW_SENSOR_ID is the placeholder, AND the file exists, preserve the file's ID.
FINAL_SENSOR_ID="$NEW_SENSOR_ID"

if [ "$NEW_SENSOR_ID" = "SENSOR_ID_PLACEHOLDER" ] && [ -f "$TARGET" ]; then
    # Extract the existing sensor_id using a lightweight python one-liner
    PARSED_ID=$(python3 -c "import json, sys; print(json.load(sys.stdin).get('sensor_id', 'SENSOR_ID_PLACEHOLDER'))" < "$TARGET" 2>/dev/null || echo "SENSOR_ID_PLACEHOLDER")
    if [ "$PARSED_ID" != "SENSOR_ID_PLACEHOLDER" ] && [ -n "$PARSED_ID" ]; then
        FINAL_SENSOR_ID="$PARSED_ID"
        echo "Found existing sensor_id: $FINAL_SENSOR_ID. Preserving it."
    fi
fi

cat > "$TARGET" <<EOF
{
  "sensor_id": "$FINAL_SENSOR_ID",
  "timezone": "Asia/Kolkata",
  "default_arguments": {
    "hef_path": "$HOME/hailo_ai_acts/resources/models/hailo8l/model.hef",
    "labels-json": "$HOME/hailo_ai_acts/resources/models/hailo8l/labels.json"
  },
  "dashboard_connectivity":{
    "api":1,
    "kafka":1
  },
  "available_activities": [
    "traffic_overspeeding_distancewise",
    "TimeBasedUnauthorizedAccess",
    "UnsafeZone",
    "PPE",
    "WAH",
    "CameraTampering"
  ],
  "active_activities": [
    "PPE",
    "WAH",
    "CameraTampering"
  ],
  "radar_config": {
    "port": "/dev/ttyACMr",
    "baudrate": 9600,
    "max_age": 10,
    "max_diff_rais": 10
  },
  "camera_details": {
    "RTSP_URL": "rtsp://admin:Arresto%402024@172.16.16.151:554/video/live?channel=1%26subtype=0",
    "camera_ip": "172.16.16.151",
    "username": "admin",
    "password": "Arresto%402024",
    "IF_USB_CAMERA": "False",
    "USB_CAM_INPUT": ""
  },
  "activities_data": {
    "traffic_overspeeding_distancewise": {
      "zones": {
        "zone1": [
          [28, 942],
          [2643, 1037],
          [2620, 1177],
          [39, 1054]
        ]
      },
      "parameters": {
        "real_distance": 10,
        "speed_limit": {
          "car": 30,
          "truck": 30,
          "motorcycle": 30
        },
        "calibration": 1.3668,
        "radar_calibration": 1,
        "lines": {
          "lane1": [
            [1304, 925],
            [1232, 1110]
          ]
        }
      }
    },
    "TimeBasedUnauthorizedAccess":{
      "zones":{
        "zone1":[[0, 0], [1080, 0], [1080, 720], [0, 720]]
      },
      "parameters":{
        "subcategory_mapping":["person"],
        "frame_accuracy": 2,
        "relay":0,
        "switch_relay":[1,2],
        "scheduled_time": [{
          "time_start_end":[["03:00","17:00"],["18:00","18:30"]],
          "days": ["Monday", "Wednesday", "Friday"]
        }],
        "last_check_time":0
      }
    },
    "UnsafeZone":{
      "zones":{
        "zone1":[[0, 0], [1080, 0], [1080, 720], [0, 720]]
      },
      "parameters":{
        "subcategory_mapping":["person"],
        "relay":0,
        "switch_relay":[1,2],
        "frame_accuracy": 2,
        "last_check_time":0
      }
    },
    "PPE":{
      "zones":{
        "zone1":[[0, 0], [1080, 0], [1080, 720], [0, 720]]
      },
      "parameters":{
        "subcategory_mapping":{"head":"No Helmet","no-vest":"No Vest"},
        "switch_relay":[1,2],
        "frame_accuracy": 5,
        "last_check_time":0,
        "relay":0
      }
    },
    "WAH":{
      "zones":{
        "zone1":[[0, 0], [1080, 0], [1080, 720], [0, 720]]
      },
      "parameters":{
        "frame_accuracy": 100,
        "subcategory_mapping":{"harness":"Harness","hooks":"Hooks"},
        "condition_label":["scaffolding","bricks"],
        "missing_subcategory":"No Harness or Hooks",
        "no_zone":1,
        "relay":0,
        "switch_relay":[1,2],
        "last_check_time":0,
        "yoloe":1,
        "yoloe_interval":300,
        "yoloe_confidence":0.1,
        "yoloe_max_stale_age":60
      }
    },
    "CameraTampering":{
      "zones":{
        "zone1":[[0, 0], [1080, 0], [1080, 720], [0, 720]]
      },
      "parameters":{
        "frame_accuracy": 2,
        "alert_interval": 30,
        "motion_thres":0.65,
        "brightness_thres":60,
        "variance_thres":15,
        "entropy_thres":1.5,
        "edge_ratio_thres":0.008,
        "hash_diff_thres":20,
        "relay":0,
        "switch_relay":[1,2],
        "last_check_time":0
      }
    }
  },
  "save_settings": {
    "save_snapshots": 0,
    "save_rtsp_images": 0,
    "take_cgi_snapshots": 0,
    "take_video":1
  },
  "calibration_required": 1
}
EOF

echo "Wrote configuration to $TARGET"

