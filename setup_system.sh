#!/bin/bash
set -euo pipefail

# =============================================================================
# Raspberry Pi 5 - Hailo System Preparation
# =============================================================================
# Prepares Raspberry Pi 5 for running Hailo AI applications

echo "🚀 Starting Raspberry Pi 5 Hailo System Preparation..."

# =============================================================================
# 1. Update OS
# =============================================================================

echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y
sudo apt install -y jq


# =============================================================================
# 2. Install Essential Dependencies
# =============================================================================

echo "🔧 Installing essential dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    libusb-1.0-0-dev \
    libudev-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    git \
    curl \
    wget \
    build-essential \
    jq

# =============================================================================
# 3. Install Hailo Packages
# =============================================================================

echo "🤖 Installing Hailo packages..."

# Install Hailo packages
sudo apt install -y hailo-all

# Check if HailoRT driver is available
echo "🔍 Checking HailoRT driver installation..."
if ! lsmod | grep -q "hailo" && ! command -v hailortcli &> /dev/null; then
    echo "⚠️  HailoRT driver not found, installing manually..."
    
    # Install build dependencies
    sudo apt install -y build-essential linux-headers-$(uname -r) raspberrypi-kernel-headers
    
    # Clone and build HailoRT drivers
    echo "📥 Cloning HailoRT drivers..."
    git clone --depth 1 -b v4.20.0 https://github.com/hailo-ai/hailort-drivers.git
    cd hailort-drivers/linux/pcie
    
    echo "🔨 Building HailoRT drivers..."
    make clean && make all
    sudo make install
    
    # Install udev rules
    sudo install -m644 51-hailo-udev.rules /etc/udev/rules.d/
    
    # Update kernel modules
    sudo depmod -a
    
    # Reload udev rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
    
    # Load Hailo PCI module
    sudo modprobe hailo_pci
    
    cd ../..
    echo "✅ HailoRT drivers installed manually"
else
    echo "✅ HailoRT drivers already available"
fi

# =============================================================================
# 4. PCIe Gen 3 Enablement (Raspberry Pi 5)
# =============================================================================

echo "🔌 Enabling PCIe Gen 3 speed..."

# Enable PCIe Gen 3 in config.txt (newer Raspberry Pi OS)
if ! grep -q "pcie_gen3=1" /boot/firmware/config.txt; then
    echo "pcie_gen3=1" | sudo tee -a /boot/firmware/config.txt
    echo "✅ PCIe Gen 3 enabled in /boot/firmware/config.txt"
else
    echo "✅ PCIe Gen 3 already enabled"
fi

# =============================================================================
# 5. Device Access Setup
# =============================================================================

echo "🔌 Setting up device access..."



# Create udev rules for radar devices
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-radar.rules
# Radar device rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666",SYMLINK+="ttyACMr"
EOF

# Create udev rules for relay devices
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-relay.rules
# USB Relay device rules - Multiple vendor support
# V-USB relays (common open-source)
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05df", MODE="0666", GROUP="plugdev", SYMLINK+="relay_device"
# Microchip USB relays
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="f5fe", MODE="0666", GROUP="plugdev"
# Generic HID devices (for compatibility)
SUBSYSTEM=="hidraw", MODE="0666", GROUP="plugdev"
# USB subsystem rules for relay devices
SUBSYSTEM=="usb", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05df", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="f5fe", MODE="0666", GROUP="plugdev"
EOF


# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to necessary groups for device access
sudo usermod -a -G video,dialout,plugdev,gpio,i2c,spi $USER

echo "✅ Device access configured (Radar, Relay, GPIO)"

# =============================================================================
# 6. Verification
# =============================================================================

echo "🔍 Verifying installation..."

# Check HailoRT
if command -v hailortcli &> /dev/null; then
    echo "✅ HailoRT CLI available"
else
    echo "❌ HailoRT CLI not found"
fi

# Check TAPPAS
if dpkg -l | grep -q hailo-tappas-core; then
    echo "✅ Hailo TAPPAS installed"
else
    echo "❌ Hailo TAPPAS not found"
fi

# Check GStreamer
if command -v gst-launch-1.0 &> /dev/null; then
    echo "✅ GStreamer available"
else
    echo "❌ GStreamer not found"
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python3 available"

    # Check if pip can install hidapi
    if python3 -c "import pip" &> /dev/null; then
        echo "💡 Tip: Install HID library for USB relay support:"
        echo "   pip3 install hidapi"
    fi
else
    echo "❌ Python3 not found"
fi

# Check USB/HID device access
echo ""
echo "🔌 Checking USB device access..."
if ls /dev/hidraw* 1> /dev/null 2>&1; then
    echo "✅ HID devices found: $(ls /dev/hidraw* | wc -l) device(s)"
    ls -la /dev/hidraw* | head -3
else
    echo "ℹ️  No HID devices currently connected (normal if relay not plugged in)"
fi

# Check user groups for device access
if groups $USER | grep -q -E "(dialout|plugdev)"; then
    echo "✅ User has USB device access groups"
else
    echo "⚠️  User may need additional groups for USB device access"
fi

# =============================================================================
# 7. Final Instructions
# =============================================================================

echo
echo "🎉 Raspberry Pi 5 Hailo preparation completed!"
echo
echo "Your system is now ready to run Hailo AI applications."
echo
echo "Next steps:"
echo "1. Reboot your system: sudo reboot"
echo "2. Connect your Hailo device"
echo "3. Run your Hailo applications"
echo
echo "Happy AI development! 🚀" 