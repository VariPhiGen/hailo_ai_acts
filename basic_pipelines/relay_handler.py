import hid
import atexit
import time


class Relay(object):
    """Relay Controller using HID interface (RPi safe)"""

    def __init__(self, idVendor=0x16c0, idProduct=0x05df):
        self.vendor_id = idVendor
        self.product_id = idProduct
        self.device = None

        self.start_time = {i: 0 for i in range(1, 9)}
        self.auto_off_sec = 3
        self.state_cache = {i: False for i in range(1, 9)}

    # -----------------------------
    # INIT (FIXED)
    # -----------------------------
    def initiate_relay(self):
        try:
            for d in hid.enumerate():
                if d["vendor_id"] == self.vendor_id and d["product_id"] == self.product_id:
                    self.device = hid.device()
                    self.device.open_path(d["path"])   # ✅ ONLY correct open
                    self.device.set_nonblocking(1)
                    atexit.register(self.cleanup)

                    print("✅ Relay initialized correctly")
                    print(f"   Manufacturer: {d.get('manufacturer_string')}")
                    print(f"   Product: {d.get('product_string')}")
                    return True

            raise RuntimeError("Relay HID device not found")

        except Exception as e:
            print(f"❌ Failed to initialize relay: {e}")
            return False

    def cleanup(self):
        if self.device:
            self.device.close()
            print("🧹 Relay connection closed")

    # -----------------------------
    # RELAY CONTROL (FIXED)
    # -----------------------------
    def state(self, relay, on=None):
        """
        Setter-only (reliable on Linux)

        relay: 1–8 or 0 (all)
        on: True / False
        """

        if on is None:
            # Return cached state (since reading is unreliable)
            return self.state_cache if relay == 0 else self.state_cache[relay]

        if relay == 0:
            self.device.write([0x00, 0xFE] if on else [0x00, 0xFC])
            for i in self.state_cache:
                self.state_cache[i] = on
        else:
            self.device.write([0x00, 0xFF, relay] if on else [0x00, 0xFD, relay])
            self.state_cache[relay] = on

        if on and relay != 0:
            self.start_time[relay] = time.time()

    # -----------------------------
    # AUTO-OFF (UNCHANGED)
    # -----------------------------
    def check_auto_off(self, relays_to_check):
        for relay in relays_to_check:
            start = self.start_time.get(relay, 0)
            if start and (time.time() - start) > self.auto_off_sec:
                print(f"⏱️ Auto-off triggered for relay {relay}")
                self.state(relay, False)
                self.start_time[relay] = 0
