#!/bin/bash
# =============================================================================
# run_detection_debug.sh — Debug mode detection launcher
#
# Differences from run_detection_with_display.sh:
#   - Sets DEBUG_MODE=1 so detection.py emits verbose YOLOE / Kafka logs
#   - No retry loop — crashes are shown immediately without waiting 60 seconds
#   - Output is NOT suppressed; all DEBUG: lines flow to stdout
#
# Usage:
#   bash run_detection_debug.sh [config_file]   (default: configuration.json)
# =============================================================================
set -euo pipefail

CONFIG_FILE="${1:-configuration.json}"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status()  { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
print_success() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅${NC} $1"; }
print_warning() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"; }
print_error()   { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌${NC} $1"; }
print_debug()   { echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] 🔍${NC} $1"; }

# ── Read camera input ─────────────────────────────────────────────────────────
read_camera_input_from_config() {
    local config_file="$1"
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file not found: $config_file"
        return 1
    fi
    if ! command -v jq &>/dev/null; then
        print_error "jq not found. Install with: sudo apt-get install -y jq"
        return 1
    fi
    local is_usb
    is_usb=$(jq -r '.camera_details.IF_USB_CAMERA' "$config_file" 2>/dev/null)
    if [[ "$is_usb" == "True" || "$is_usb" == "true" ]]; then
        local usb_input
        usb_input=$(jq -r '.camera_details.USB_CAM_INPUT' "$config_file" 2>/dev/null)
        if [[ "$usb_input" != "null" && -n "$usb_input" ]]; then
            echo "$usb_input"; return 0
        fi
    else
        local rtsp_url
        rtsp_url=$(jq -r '.camera_details.RTSP_URL' "$config_file" 2>/dev/null)
        if [[ "$rtsp_url" != "null" && -n "$rtsp_url" ]]; then
            echo "$rtsp_url"; return 0
        fi
    fi
    print_error "No valid camera input found in $config_file"
    return 1
}

# ── Startup banner ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       AI Acts — DEBUG Detection Mode                 ║${NC}"
echo -e "${CYAN}║  YOLOE calls, Kafka sends and errors are logged here ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

print_status "Config file : $CONFIG_FILE"

CAMERA_INPUT=$(read_camera_input_from_config "$CONFIG_FILE") || exit 1
print_status "Camera input: $CAMERA_INPUT"

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [[ -f "setup_env.sh" ]]; then
    source setup_env.sh
    print_success "Environment activated"
else
    print_error "setup_env.sh not found"
    exit 1
fi

# ── Enable debug mode for detection.py / yoloe_handler.py / kafka_handler.py ─
export DEBUG_MODE=1
print_debug "DEBUG_MODE=1 — verbose YOLOE + Kafka logging enabled"
echo ""

print_status "▶  Starting detection.py in debug mode..."
echo ""

# Run once — no retry loop so the full traceback is immediately visible
python basic_pipelines/detection.py \
    --i "$CAMERA_INPUT" \
    --disable-sync \
    2>&1 | tee /tmp/hailo_debug_$(date '+%Y%m%d_%H%M%S').log

print_status "Detection process exited. See /tmp/hailo_debug_*.log for the full run log."
