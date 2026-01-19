#!/bin/bash

# Determine the directory of the script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Define the target directory relative to the script's location
TARGET_DIR="$SCRIPT_DIR/resources/models/hailo8"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of Hailo-8 model URLs
H8_HEFS=(
  "https://variphi.s3.ap-south-1.amazonaws.com/hailo-model/base.hef"
  "https://variphi.s3.ap-south-1.amazonaws.com/hailo-model/ppe-labels.json"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov5m_wo_spp.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov11n.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov11s.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8s.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov6n.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/scdepthv3.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8s_pose.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov5n_seg.hef"
)


# Download each model
for model_url in "${H8_HEFS[@]}"; do
  model_name=$(basename "$model_url")
  model_path="$TARGET_DIR/$model_name"

  # Check if the model already exists
  if [ -f "$model_path" ]; then
    echo "Model '$model_name' already exists. Skipping download."
  else
    echo "Downloading '$model_name'..."
    curl -L -o "$model_path" "$model_url"
    if [ $? -eq 0 ]; then
      echo "Downloaded '$model_name' successfully."
    else
      echo "Failed to download '$model_name'."
    fi
  fi
done
