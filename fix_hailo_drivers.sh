#!/bin/bash
set -euo pipefail

# =============================================================================
# Hailo Driver Fix Script for Raspberry Pi 5
# =============================================================================
# This script manually installs HailoRT drivers when standard package installation fails

echo "🔧 Starting Hailo Driver Fix for Raspberry Pi 5..."

# =============================================================================
# 1. Check Current Status
# =============================================================================

echo "🔍 Checking current Hailo driver status..."

if lsmod | grep -q "hailo" && command -v hailortcli &> /dev/null; then
    echo "✅ Hailo drivers appear to be working correctly"
    echo "Current Hailo modules loaded:"
    lsmod | grep hailo || echo "No Hailo modules currently loaded"
    exit 0
fi

echo "⚠️  Hailo drivers not found or not working properly"
echo "Proceeding with manual installation..."

# =============================================================================
# 2. Install Build Dependencies
# =============================================================================

echo "📦 Installing build dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    linux-headers-$(uname -r) \
    raspberrypi-kernel-headers \
    git \
    make

# =============================================================================
# 3. Clone HailoRT Drivers
# =============================================================================

echo "📥 Cloning HailoRT drivers repository..."

# Remove existing directory if it exists
if [ -d "hailort-drivers" ]; then
    echo "🗑️  Removing existing hailort-drivers directory..."
    rm -rf hailort-drivers
fi

# Clone the drivers
git clone --depth 1 -b v4.20.0 https://github.com/hailo-ai/hailort-drivers.git

if [ ! -d "hailort-drivers" ]; then
    echo "❌ Failed to clone HailoRT drivers repository"
    exit 1
fi

echo "✅ HailoRT drivers repository cloned successfully"

# =============================================================================
# 4. Build and Install Drivers
# =============================================================================

echo "🔨 Building and installing HailoRT drivers..."

cd hailort-drivers/linux/pcie

# Clean previous builds
echo "🧹 Cleaning previous builds..."
make clean

# Build the drivers
echo "🔨 Building drivers..."
make all

if [ $? -ne 0 ]; then
    echo "❌ Failed to build HailoRT drivers"
    exit 1
fi

# Install the drivers
echo "📦 Installing drivers..."
sudo make install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install HailoRT drivers"
    exit 1
fi

echo "✅ Drivers built and installed successfully"

# =============================================================================
# 5. Install UDEV Rules
# =============================================================================

echo "📋 Installing UDEV rules..."

if [ -f "51-hailo-udev.rules" ]; then
    sudo install -m644 51-hailo-udev.rules /etc/udev/rules.d/
    echo "✅ UDEV rules installed"
else
    echo "⚠️  UDEV rules file not found, skipping..."
fi

# =============================================================================
# 6. Update Kernel Modules
# =============================================================================

echo "🔧 Updating kernel modules..."
sudo depmod -a

# =============================================================================
# 7. Reload UDEV Rules
# =============================================================================

echo "🔄 Reloading UDEV rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# =============================================================================
# 8. Load Hailo PCI Module
# =============================================================================

echo "🚀 Loading Hailo PCI module..."
sudo modprobe hailo_pci

if [ $? -ne 0 ]; then
    echo "⚠️  Failed to load hailo_pci module (this might be normal if no device is connected)"
else
    echo "✅ Hailo PCI module loaded successfully"
fi

# =============================================================================
# 9. Verification
# =============================================================================

echo "🔍 Verifying installation..."

# Check if modules are available
if lsmod | grep -q "hailo"; then
    echo "✅ Hailo modules are loaded"
    lsmod | grep hailo
else
    echo "ℹ️  Hailo modules not loaded (normal if no device connected)"
fi

# Check if hailortcli is available
if command -v hailortcli &> /dev/null; then
    echo "✅ HailoRT CLI is available"
else
    echo "⚠️  HailoRT CLI not found - you may need to install Hailo packages separately"
fi

# =============================================================================
# 10. Cleanup
# =============================================================================

echo "🧹 Cleaning up build files..."
cd ../..
rm -rf hailort-drivers

# =============================================================================
# 11. Final Instructions
# =============================================================================

echo
echo "🎉 Hailo driver installation completed!"
echo
echo "Next steps:"
echo "1. Connect your Hailo device"
echo "2. Reboot your system: sudo reboot"
echo "3. After reboot, check if drivers are working:"
echo "   - lsmod | grep hailo"
echo "   - hailortcli scan"
echo
echo "If you still have issues after reboot, try:"
echo "   - sudo modprobe hailo_pci"
echo "   - Check dmesg for any error messages"
echo
echo "Happy AI development! 🚀" 