#!/bin/bash
set -euo pipefail

# Configuration
CONFIG_FILE="${1:-configuration.json}"
MAX_RETRIES=999999  # Run indefinitely
MAX_FAILED_ATTEMPTS=10  # Reboot after 10 failed attempts
SLEEP_DELAYS=(60 300 600)  # 1 min, 5 min, 10 min in seconds
DELAY_INDEX=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

# Function to read RTSP link from configuration.json
read_rtsp_from_config() {
    local config_file="$1"
    
    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file not found: $config_file"
        return 1
    fi
    
    # Check if jq is available for JSON parsing
    if command -v jq &> /dev/null; then
        # Use jq for reliable JSON parsing
        local rtsp_url=$(jq -r '.camera_details.RTSP_URL' "$config_file" 2>/dev/null)
        if [[ "$rtsp_url" != "null" && -n "$rtsp_url" ]]; then
            echo "$rtsp_url"
            return 0
        fi
    else
        # Fallback to grep/sed for basic parsing
        local rtsp_url=$(grep -o '"RTSP_URL"[[:space:]]*:[[:space:]]*"[^"]*"' "$config_file" | sed 's/.*"RTSP_URL"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
        if [[ -n "$rtsp_url" ]]; then
            echo "$rtsp_url"
            return 0
        fi
    fi
    
    print_error "RTSP_URL not found in configuration file: $config_file"
    return 1
}

# Read RTSP link from configuration
print_status "Reading RTSP link from configuration file: $CONFIG_FILE"
RTSP_LINK=$(read_rtsp_from_config "$CONFIG_FILE")
if [[ $? -ne 0 ]]; then
    print_error "Failed to read RTSP link from configuration"
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configuration.json"
    echo ""
    echo "Make sure your configuration.json contains:"
    echo '  "camera_details": {'
    echo '    "RTSP_URL": "rtsp://username:password@camera-ip:554/stream"'
    echo '  }'
    exit 1
fi

print_status "Starting SVDS Detection Pipeline with continuous auto-restart"
print_status "Config File: $CONFIG_FILE"
print_status "RTSP Link: $RTSP_LINK"
print_status "Max retries: $MAX_RETRIES (continuous operation)"

# Function to run detection
run_detection() {
    local attempt=$1
    print_status "Attempt $attempt: Starting detection pipeline..."
    
    # Change to SVDS directory
    #cd SVDS || {
    #    print_error "Failed to change to SVDS directory"
    #    return 1
    #}
    
    # Source the environment
    if [[ -f "setup_env.sh" ]]; then
        source setup_env.sh
        print_success "Environment activated"
    else
        print_error "setup_env.sh not found"
        return 1
    fi
    
    # Run the detection pipeline
    print_status "Running: python basic_pipelines/detection.py --i $RTSP_LINK --disable-sync --disable-display"
    python basic_pipelines/detection.py --i $RTSP_LINK --disable-sync --disable-display
}

# Function to get sleep delay
get_sleep_delay() {
    local delay=${SLEEP_DELAYS[$DELAY_INDEX]}
    if [[ $DELAY_INDEX -lt $((${#SLEEP_DELAYS[@]} - 1)) ]]; then
        DELAY_INDEX=$((DELAY_INDEX + 1))
    fi
    echo $delay
}

# Main loop
attempt=1
failed_attempts=0
while [[ $attempt -le $MAX_RETRIES ]]; do
    print_status "=== Starting Detection Pipeline (Attempt $attempt) ==="
    
    # Run detection
    if run_detection $attempt; then
        print_success "Detection pipeline completed successfully"
        break
    else
        failed_attempts=$((failed_attempts + 1))
        print_warning "Detection pipeline exited with error (Attempt $attempt, Failed: $failed_attempts/$MAX_FAILED_ATTEMPTS)"
        
        # Check if we should reboot
        if [[ $failed_attempts -ge $MAX_FAILED_ATTEMPTS ]]; then
            print_error "CRITICAL: $failed_attempts failed attempts reached. Rebooting system in 30 seconds..."
            print_error "This will attempt to resolve persistent system issues."
            
            # Countdown to reboot
            for ((i=30; i>0; i--)); do
                print_error "Rebooting in $i seconds... (Press Ctrl+C to cancel)"
                sleep 1
            done
            
            print_error "Rebooting system now..."
            sudo reboot
            exit 0
        fi
        
        # Get sleep delay
        sleep_time=$(get_sleep_delay)
        print_status "Waiting ${sleep_time} seconds before retry..."
        
        # Sleep with progress indicator
        for ((i=1; i<=sleep_time; i++)); do
            if [[ $((i % 30)) -eq 0 ]]; then
                print_status "Waiting... ${i}/${sleep_time} seconds"
            fi
            sleep 1
        done
        
        print_status "Retrying..."
        attempt=$((attempt + 1))
    fi
done

print_success "Detection pipeline finished"
