# If Driver is missing for 4.20 Host
sudo apt install build-essential linux-headers-$(uname -r) raspberrypi-kernel-headers
git clone --depth 1 -b v4.20.0 https://github.com/hailo-ai/hailort-drivers.git
cd hailort-drivers/linux/pcie
make clean && make all
sudo make install
sudo install -m644 51-hailo-udev.rules /etc/udev/rules.d/
sudo depmod -a
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo modprobe hailo_pci

# If New Version is Not working Downgrade it 
sudo apt install hailo-tappas-core=3.30.0-1 hailort=4.19.0-3 hailo-dkms=4.19.0-1 python3-hailort=4.19.0-2
