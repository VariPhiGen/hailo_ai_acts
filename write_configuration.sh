#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-configuration.json}"

cat > "$TARGET" <<EOF
{
  "sensor_id": "SENSOR_ID_PLACEHOLDER",
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
    "PPE"
  ],
  "active_activities": [
    "PPE"
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
        "relay":1,
        "switch_relay":[1,2],
        "scheduled_time": [{
          "time_start_end":[["03:00","17:00"],["18:00","18:30"]],
          "days": ["Monday", "Wednesday", "Friday"]
        }],
        "timezone":"Asia/Tokyo",
        "last_check_time":0
      }
    },
    "UnsafeZone":{
      "zones":{
        "zone1":[[0, 0], [1080, 0], [1080, 720], [0, 720]]
      },
      "parameters":{
        "subcategory_mapping":["person"],
        "relay":1,
        "switch_relay":[1,2],
        "frame_accuracy": 2,
        "timezone":"Asia/Tokyo",
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
        "timezone":"Asia/Kolkata",
        "last_check_time":0,
        "relay":0
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
        "relay":1,
        "switch_relay":[1,2],
        "timezone":"Asia/Kolkata",
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

