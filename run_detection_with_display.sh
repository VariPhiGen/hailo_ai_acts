#!/bin/bash
set -euo pipefail

# Configuration
CONFIG_FILE="${1:-configuration.json}"
MAX_RETRIES=999999  # Run indefinitely
MAX_FAILED_ATTEMPTS=10  # Reboot after 10 failed attempts
SLEEP_DELAYS=(60 300 600)  # 1 min, 5 min, 10 min
DELAY_INDEX=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status()   { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
print_success()  { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅${NC} $1"; }
print_warning()  { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"; }
print_error()    { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌${NC} $1"; }

# Function to read camera input (RTSP or USB)
read_camera_input_from_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file not found: $config_file"
        return 1
    fi

    if command -v jq &> /dev/null; then
        local is_usb=$(jq -r '.camera_details.IF_USB_CAMERA' "$config_file" 2>/dev/null)
        if [[ "$is_usb" == "True" || "$is_usb" == "true" ]]; then
            local usb_input=$(jq -r '.camera_details.USB_CAM_INPUT' "$config_file" 2>/dev/null)
            if [[ "$usb_input" != "null" && -n "$usb_input" ]]; then
                echo "$usb_input"
                return 0
            fi
        else
            local rtsp_url=$(jq -r '.camera_details.RTSP_URL' "$config_file" 2>/dev/null)
            if [[ "$rtsp_url" != "null" && -n "$rtsp_url" ]]; then
                echo "$rtsp_url"
                return 0
            fi
        fi
    else
        print_error "jq not found, please install it: sudo apt-get install -y jq"
        return 1
    fi

    print_error "No valid camera input found in $config_file"
    return 1
}

print_status "Reading camera input from $CONFIG_FILE"
CAMERA_INPUT=$(read_camera_input_from_config "$CONFIG_FILE")
if [[ $? -ne 0 ]]; then
    print_error "Failed to read camera input"
    exit 1
fi

print_status "Starting SVDS Detection Pipeline"
print_status "Config File: $CONFIG_FILE"
print_status "Camera Input: $CAMERA_INPUT"

# Function to run detection
run_detection() {
    local attempt=$1
    print_status "Attempt $attempt: Starting detection pipeline..."

    if [[ -f "setup_env.sh" ]]; then
        source setup_env.sh
        print_success "Environment activated"
    else
        print_error "setup_env.sh not found"
        return 1
    fi

    print_status "Running: python basic_pipelines/detection.py --i $CAMERA_INPUT --disable-sync"
    python basic_pipelines/detection.py --i "$CAMERA_INPUT" --disable-sync
}

# Retry logic
get_sleep_delay() {
    local delay=${SLEEP_DELAYS[$DELAY_INDEX]}
    if [[ $DELAY_INDEX -lt $((${#SLEEP_DELAYS[@]} - 1)) ]]; then
        DELAY_INDEX=$((DELAY_INDEX + 1))
    fi
    echo $delay
}

attempt=1
failed_attempts=0
while [[ $attempt -le $MAX_RETRIES ]]; do
    print_status "=== Starting Detection Pipeline (Attempt $attempt) ==="

    if run_detection $attempt; then
        print_success "Detection pipeline completed successfully"
        break
    else
        failed_attempts=$((failed_attempts + 1))
        print_warning "Detection failed (Attempt $attempt, Failed: $failed_attempts/$MAX_FAILED_ATTEMPTS)"

        if [[ $failed_attempts -ge $MAX_FAILED_ATTEMPTS ]]; then
            print_error "CRITICAL: $failed_attempts failed attempts reached. Rebooting system..."
            sudo reboot
            exit 0
        fi

        sleep_time=$(get_sleep_delay)
        print_status "Waiting $sleep_time seconds before retry..."
        sleep $sleep_time
        attempt=$((attempt + 1))
    fi
done

print_success "Detection pipeline finished"
