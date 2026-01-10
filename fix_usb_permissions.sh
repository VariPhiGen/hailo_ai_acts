#!/bin/bash

echo "🔧 USB Relay Permission Fix Script"
echo "=================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Don't run this script as root! It will add your user to groups."
    exit 1
fi

USERNAME=$(whoami)
echo "👤 Running as user: $USERNAME"

echo ""
echo "1️⃣ Adding user to USB groups..."
sudo usermod -a -G dialout,plugdev $USERNAME

echo ""
echo "2️⃣ Setting permissions on HID devices..."
sudo chmod 666 /dev/hidraw* 2>/dev/null || echo "⚠️ No HID devices found or permission denied"

echo ""
echo "3️⃣ Creating udev rule for USB relays..."
sudo tee /etc/udev/rules.d/99-usb-relay.rules > /dev/null <<EOF
# USB Relay devices - allow user access
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="05df", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="f5fe", MODE="0666"
EOF

echo ""
echo "4️⃣ Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "✅ USB permissions configured!"
echo ""
echo "📝 Next steps:"
echo "1. Logout and login again (or reboot)"
echo "2. Reconnect the USB relay device"
echo "3. Run the diagnostic script: python3 relay_diagnostic.py"
echo ""
echo "If issues persist, try running the detection with sudo:"
echo "sudo python3 basic_pipelines/detection.py --i /path/to/video.mp4"
