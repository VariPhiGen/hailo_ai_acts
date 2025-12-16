#!/bin/bash
set -euo pipefail

# =============================================================================
# Sequential Setup Script for Hailo AI Applications
# =============================================================================
# Runs setup scripts in order:
#   1. setup_system.sh - System preparation and Hailo driver installation
#   2. install.sh - Python environment and package installation
#   3. download_resources.sh - Download model files and resources
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check if script exists and is executable
check_script() {
    local script=$1
    if [[ ! -f "$script" ]]; then
        print_error "Script '$script' not found!"
        exit 1
    fi
    if [[ ! -x "$script" ]]; then
        print_warning "Script '$script' is not executable. Making it executable..."
        chmod +x "$script"
    fi
}

# Function to run a script with error handling
run_script() {
    local script=$1
    local step_name=$2
    shift 2
    local args=("$@")
    
    print_step "Step: $step_name"
    echo "Running: $script ${args[*]}"
    echo ""
    
    if bash "$script" "${args[@]}"; then
        print_success "$step_name completed successfully!"
        echo ""
        return 0
    else
        print_error "$step_name failed with exit code $?"
        echo ""
        print_error "Setup sequence aborted. Please fix the error and try again."
        exit 1
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     ${GREEN}🚀 Hailo AI Applications - Complete Setup Sequence${NC}                    ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root (some steps may need sudo)
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. Some steps may require user privileges."
fi

# Check all required scripts exist
echo "Checking required scripts..."
check_script "setup_system.sh"
check_script "install.sh"
check_script "download_resources.sh"
print_success "All required scripts found!"
echo ""

# Parse arguments for install.sh
INSTALL_ARGS=()
DOWNLOAD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            DOWNLOAD_ARGS+=("--all")
            shift
            ;;
        --no-installation)
            INSTALL_ARGS+=("--no-installation")
            shift
            ;;
        --pyhailort)
            INSTALL_ARGS+=("--pyhailort" "$2")
            shift 2
            ;;
        --pytappas)
            INSTALL_ARGS+=("--pytappas" "$2")
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [OPTIONS]

Runs setup scripts in sequence:
  1. setup_system.sh - System preparation
  2. install.sh - Package installation
  3. download_resources.sh - Resource downloads

Options:
  --all                  Download all resources (for download_resources.sh)
  --no-installation      Skip installation (for install.sh)
  --pyhailort PATH       Custom pyhailort path (for install.sh)
  --pytappas PATH        Custom pytappas path (for install.sh)
  -h, --help             Show this help message

Examples:
  $0                     # Run all steps with defaults
  $0 --all               # Download all resources
  $0 --no-installation   # Skip package installation step
EOF
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get start time
START_TIME=$(date +%s)

# =============================================================================
# Step 1: System Setup
# =============================================================================
run_script "setup_system.sh" "System Setup (Hailo drivers, dependencies)"

# =============================================================================
# Step 2: Installation
# =============================================================================
run_script "install.sh" "Package Installation (Python, dependencies)" "${INSTALL_ARGS[@]}"

# =============================================================================
# Step 3: Download Resources
# =============================================================================
run_script "download_resources.sh" "Resource Downloads (Models, videos)" "${DOWNLOAD_ARGS[@]}"

# =============================================================================
# Completion
# =============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                        ${GREEN}🎉 Setup Complete!${NC}                                    ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
print_success "All setup steps completed successfully!"
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Next steps:"
echo "  1. Reboot your system if required: sudo reboot"
echo "  2. Activate the virtual environment: source hailo_venv/bin/activate"
echo "  3. Or use the setup script: source setup_env.sh"
echo "  4. Run your Hailo applications!"
echo ""
echo -e "${GREEN}Happy AI development! 🚀${NC}"
echo ""

