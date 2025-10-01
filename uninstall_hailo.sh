#!/bin/bash

# Ensure the script is run with root privileges
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root."
  exit 1
fi

echo "Starting Hailo PCIe driver uninstallation..."

# 1. Stop the HailoRT service
echo "Stopping HailoRT service..."
systemctl stop hailort || echo "Failed to stop hailort service (may not be running)."
systemctl disable hailort || echo "Failed to disable hailort service."

# 2. Unload the Hailo driver
echo "Unloading Hailo PCIe driver..."
rmmod hailo_pci || echo "Failed to unload hailo_pci module (may not be loaded)."

# 3. Remove Hailo device file
echo "Removing Hailo device file..."
rm -f /dev/hailo0

# 4. Purge Hailo packages
echo "Purging Hailo packages..."
apt-get purge -y hailo* hailort hailofw hailo-tappas-core hailort-pcie-driver
apt-get autoremove -y && apt-get clean

# 5. Remove Hailo directories and configurations
echo "Removing Hailo directories and configurations..."
rm -rf /opt/hailo /usr/local/hailo ~/.hailo /lib/firmware/hailo
rm -f /etc/udev/rules.d/*hailo*.rules

# 6. Remove Hailo from DKMS (if applicable)
if dkms status | grep -q hailo_pci; then
  echo "Removing Hailo from DKMS..."
  dkms remove hailo_pci/$(dkms status | grep hailo_pci | awk '{print $2}') --all
else
  echo "Hailo not found in DKMS."
fi

# 7. Remove Hailo systemd services
echo "Removing Hailo systemd services..."
systemctl stop hailo* || echo "Failed to stop Hailo services (may not be running)."
systemctl disable hailo* || echo "Failed to disable Hailo services."
rm -f /etc/systemd/system/hailo*.service
systemctl daemon-reload

# 8. Reboot the system
echo "Uninstallation complete. Rebooting the system..."
reboot
