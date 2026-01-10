import hid
import atexit
import time

"""
This Relay object uses the HID library instead of usb. 

Some scant details about the USB Relay:
http://vusb.wikidot.com/project:driver-less-usb-relays-hid-interface

cython-hidapi module:
https://github.com/trezor/cython-hidapi

Installing the module:
sudo apt-get install python-dev libusb-1.0-0-dev libudev-dev
pip install --upgrade setuptools
pip install hidapi

A list of available methods for the hid object:
https://github.com/trezor/cython-hidapi/blob/6057d41b5a2552a70ff7117a9d19fc21bf863867/chid.pxd#L9
"""


class Relay(object):
    """Relay Controller using HID interface"""

    def __init__(self, idVendor=0x16c0, idProduct=0x05df):
        self.start_time = {i: 0 for i in range(1, 9)}
        self.auto_off_sec=3

        self.vendor_id = idVendor
        self.product_id = idProduct
        self.device = None

    def initiate_relay(self):
        try:
            print(f"🔌 Attempting to open relay device: Vendor={hex(self.vendor_id)}, Product={hex(self.product_id)}")

            self.device = hid.device()
            self.device.open(self.vendor_id, self.product_id)
            self.device.set_nonblocking(1)

            # Verify device is actually open by trying to get manufacturer string
            try:
                manufacturer = self.device.get_manufacturer_string()
                product = self.device.get_product_string()
                print(f"✅ Relay initialized successfully!")
                print(f"   Manufacturer: {manufacturer}")
                print(f"   Product: {product}")
                print(f"   Vendor ID: {hex(self.vendor_id)}, Product ID: {hex(self.product_id)}")
            except Exception as info_e:
                print(f"⚠️ Relay opened but info query failed: {info_e}")
                print("   Device may be functional despite this warning.")

            # Register cleanup on program exit
            atexit.register(self.cleanup)
            return True

        except PermissionError as e:
            print(f"❌ Permission denied accessing relay device: {e}")
            print("   Try running with sudo or check USB permissions")
            print("   On Linux: sudo usermod -a -G dialout,plugdev $USER && logout/login")
            return False
        except OSError as e:
            print(f"❌ OS error accessing relay device: {e}")
            print("   Possible causes:")
            print("   - Device not physically connected")
            print("   - Device already in use by another process")
            print("   - Wrong vendor/product ID")
            print("   - USB permissions issue")
            print("")
            print("   🔧 Quick fixes to try:")
            print("   1. Run: sudo chmod 666 /dev/hidraw*")
            print("   2. Run: ./rpi_relay_fix.sh (then logout/login)")
            print("   3. Try: sudo python3 test_relay.py")
            print("   4. Or disable relay: set 'relay': 0 in config.json")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize relay: {e}")
            print("   This could be a driver or library issue")
            return False

    def cleanup(self):
        if self.device:
            try:
                self.device.close()
                print("🧹 Relay connection closed cleanly.")
            except Exception as e:
                print(f"⚠️ Error during relay cleanup: {e}")

    def check_auto_off(self, relays_to_check):
        """
        Check auto-off for a list of relays.
        Only triggers for relays that are ON and exceeded auto_off_sec.

        relays_to_check: list of relay numbers, e.g., [1, 3, 5]
        """
        for relay in relays_to_check:
            # Only consider if relay was turned ON
            start = self.start_time.get(relay, 0)
            #print(start,relay)
            if start:
                # Check if relay is actually ON
                status = self.state(relay)  # True = ON, False = OFF
                if status:
                    # Check elapsed time
                    if (time.time() - start) > self.auto_off_sec:
                        print(f"⏱️ Auto-off triggered for relay {relay}")
                        self.state(relay, False)
                        self.start_time[relay] = 0
                elif self.start_time[relay]!=0:
                    # Relay already OFF, reset timer
                    self.start_time[relay] = 0



    def get_switch_statuses_from_report(self, report):
        """
        The report returned is an 8-int list, e.g.:
        [76, 72, 67, 88, 73, 0, 0, 2]

        The first 5 in the list are a unique ID.
        The 8th value encodes relay states in binary (reversed).
        """
        # Grab the 8th number, which is an integer
        switch_statuses = report[7]

        # Convert to binary list
        switch_statuses = [int(x) for x in list('{0:08b}'.format(switch_statuses))]

        # Reverse list: status reads right-to-left
        switch_statuses.reverse()

        return switch_statuses

    def send_feature_report(self, message):
        if not self.device:
            raise RuntimeError("Relay not initialized.")
        self.device.send_feature_report(message)

    def get_feature_report(self):
        if not self.device:
            raise RuntimeError("Relay not initialized.")
        feature = 1
        length = 8
        return self.device.get_feature_report(feature, length)

    def state(self, relay, on=None):
        """
        Getter/Setter for relay state.

        Getter:
            state(relay)
            - relay = 1–8 → returns bool
            - relay = 0 → returns list of all statuses

        Setter:
            state(relay, on=True/False)
            - relay = 1–8 or 0 (for all)
        """
        # Getter
        if on is None:
            report = self.get_feature_report()
            switch_statuses = self.get_switch_statuses_from_report(report)

            if relay == 0:
                return [bool(s) for s in switch_statuses]
            else:
                return bool(switch_statuses[relay - 1])

        # Setter
        else:
            if relay == 0:
                message = [0xFE] if on else [0xFC]
            else:
                message = [0xFF, relay] if on else [0xFD, relay]

            self.send_feature_report(message)
