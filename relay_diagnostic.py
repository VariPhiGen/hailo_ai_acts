#!/usr/bin/env python3
"""
Relay Diagnostic Script
Checks if USB relay devices are available and accessible
"""

import sys
import os

# Add the basic_pipelines directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'basic_pipelines'))

def check_hid_library():
    """Check if HID library is available"""
    try:
        import hid
        print(f"✅ HID library available: {hid.__version__}")
        return True
    except ImportError as e:
        print(f"❌ HID library not found: {e}")
        print("   Install with: pip3 install hidapi")
        return False

def check_relay_symlink():
    """Check if relay symlink exists"""
    symlink_path = "/dev/relay_device"
    if os.path.exists(symlink_path):
        print(f"✅ Relay symlink exists: {symlink_path}")
        try:
            target = os.readlink(symlink_path)
            print(f"   Points to: {target}")
            if os.path.exists(target):
                print(f"   Target exists: ✅")
            else:
                print(f"   Target missing: ❌")
        except Exception as e:
            print(f"   Error reading symlink: {e}")
        return True
    else:
        print(f"❌ Relay symlink missing: {symlink_path}")
        return False

def list_hid_devices():
    """List all available HID devices"""
    try:
        import hid
        devices = hid.enumerate()
        print(f"📋 Found {len(devices)} HID devices:")

        relay_devices = []
        for device in devices:
            vendor_id = device['vendor_id']
            product_id = device['product_id']
            manufacturer = device.get('manufacturer_string', 'Unknown')
            product = device.get('product_string', 'Unknown')

            print(f"   {hex(vendor_id)}:{hex(product_id)} - {manufacturer} - {product}")

            # Check for common USB relay devices
            if (vendor_id == 0x16c0 and product_id == 0x05df) or \
               (vendor_id == 0x04d8 and product_id == 0xf5fe) or \
               'relay' in (manufacturer + product).lower():
                relay_devices.append((vendor_id, product_id, manufacturer, product))

        if relay_devices:
            print(f"🎯 Found {len(relay_devices)} potential relay device(s):")
            for vendor_id, product_id, manufacturer, product in relay_devices:
                print(f"   {hex(vendor_id)}:{hex(product_id)} - {manufacturer} - {product}")
        else:
            print("❌ No relay devices found")
            print("   Common relay devices:")
            print("   - 16c0:05df - V-USB relays")
            print("   - 04d8:f5fe - Microchip USB relays")

        return relay_devices

    except Exception as e:
        print(f"❌ Error enumerating HID devices: {e}")
        return []

def test_relay_connection(vendor_id=0x16c0, product_id=0x05df):
    """Test connecting to a specific relay device"""
    try:
        import hid
        print(f"🔌 Testing connection to {hex(vendor_id)}:{hex(product_id)}")

        device = hid.device()
        device.open(vendor_id, product_id)
        device.set_nonblocking(1)

        print("✅ Successfully opened relay device!")

        # Try to get device info
        try:
            manufacturer = device.get_manufacturer_string()
            product = device.get_product_string()
            print(f"   Manufacturer: {manufacturer}")
            print(f"   Product: {product}")
        except Exception as info_e:
            print(f"⚠️ Device opened but info query failed: {info_e}")

        device.close()
        print("✅ Device closed successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to connect to relay device: {e}")
        print("   Possible causes:")
        print("   - Device not physically connected")
        print("   - Wrong vendor/product ID")
        print("   - Insufficient USB permissions")
        print("   - Device already in use")
        return False

def check_usb_permissions():
    """Check USB device permissions"""
    import subprocess
    try:
        # Check if user can access USB devices
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ lsusb command works (USB access available)")
        else:
            print("❌ lsusb command failed")
    except Exception as e:
        print(f"⚠️ Could not check USB permissions: {e}")

def check_user_groups():
    """Check if user has necessary groups"""
    username = os.getlogin()
    try:
        import subprocess
        result = subprocess.run(['groups', username], capture_output=True, text=True)
        groups = result.stdout.strip().split()
        print(f"👤 User {username} is in groups: {groups}")

        required_groups = ['dialout', 'plugdev']
        missing_groups = []
        for group in required_groups:
            if group not in groups:
                missing_groups.append(group)

        if missing_groups:
            print(f"⚠️ Missing groups for USB access: {missing_groups}")
            print("   Run: sudo usermod -a -G dialout,plugdev $USER && logout/login")
        else:
            print("✅ User has required USB groups")

    except Exception as e:
        print(f"⚠️ Could not check user groups: {e}")

def main():
    print("🔍 Relay Diagnostic Tool")
    print("=" * 40)

    # Check HID library
    if not check_hid_library():
        return

    # Check relay symlink
    print("\n" + "=" * 40)
    check_relay_symlink()

    # List all HID devices
    relay_devices = list_hid_devices()

    # Test default relay connection
    print("\n" + "=" * 40)
    test_relay_connection()

    # Test any found relay devices
    for vendor_id, product_id, manufacturer, product in relay_devices:
        print(f"\n" + "=" * 40)
        print(f"Testing {manufacturer} - {product}")
        test_relay_connection(vendor_id, product_id)

    # Check USB permissions
    print("\n" + "=" * 40)
    check_usb_permissions()

    # Check user groups
    print("\n" + "=" * 40)
    check_user_groups()

    print("\n" + "=" * 40)
    print("📝 Troubleshooting tips:")
    print("1. Ensure USB relay is physically connected and powered")
    print("2. Try a different USB port")
    print("3. Check USB cable and power adapter")
    print("4. Run: sudo chmod 666 /dev/hidraw*")
    print("5. Run: ./rpi_relay_fix.sh (then logout/login)")
    print("6. Check device with: lsusb | grep -i relay")
    print("7. Test with: sudo python3 test_relay.py")

if __name__ == "__main__":
    main()
