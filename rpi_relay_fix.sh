#!/bin/bash

echo "🐾 Raspberry Pi USB Relay Fix"
echo "=============================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️ This doesn't appear to be a Raspberry Pi"
    echo "   Script will continue but may not be appropriate"
fi

echo "👤 Current user: $(whoami)"

# Step 1: Add user to necessary groups
echo ""
echo "1️⃣ Adding user to USB and GPIO groups..."
sudo usermod -a -G dialout,plugdev,gpio,i2c,spi $(whoami)

# Step 2: Fix HID device permissions
echo ""
echo "2️⃣ Setting HID device permissions..."
sudo chmod 666 /dev/hidraw* 2>/dev/null || echo "   No HID devices currently connected"

# Step 3: Create comprehensive udev rules for Raspberry Pi
echo ""
echo "3️⃣ Creating udev rules for USB relays..."
sudo tee /etc/udev/rules.d/99-usb-relay.rules > /dev/null <<EOF
# USB Relay devices - allow user access
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05df", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="f5fe", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05df", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="f5fe", MODE="0666", GROUP="plugdev"

# Generic HID devices
SUBSYSTEM=="hidraw", MODE="0666", GROUP="plugdev"
EOF

# Step 4: Reload udev rules
echo ""
echo "4️⃣ Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# Step 5: Check current device status
echo ""
echo "5️⃣ Checking current USB devices..."
lsusb | grep -i relay || echo "   No relay devices found in lsusb output"

echo ""
echo "6️⃣ Checking HID devices..."
ls -la /dev/hidraw* 2>/dev/null || echo "   No HID devices found"

echo ""
echo "✅ Raspberry Pi USB relay permissions configured!"
echo ""
echo "📝 Next steps:"
echo "1. Reconnect the USB relay device to a different USB port"
echo "2. Wait 10 seconds for device to be recognized"
echo "3. Test with: python3 test_relay.py"
echo ""
echo "If still failing, try running detection with sudo:"
echo "   sudo python3 basic_pipelines/detection.py --i /path/to/video.mp4"
echo ""
echo "Or disable relay features by setting 'relay': 0 in configuration.json"
