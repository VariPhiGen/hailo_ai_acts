#!/usr/bin/env python3
"""
Simple relay test script - run with sudo if needed
"""

import sys
import os

# Add basic_pipelines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'basic_pipelines'))

try:
    from relay_handler import Relay
    print("✅ Imported relay_handler successfully")
except ImportError as e:
    print(f"❌ Failed to import relay_handler: {e}")
    sys.exit(1)

def test_relay():
    print("🔌 Testing USB Relay Connection")
    print("=" * 40)

    # Try default relay
    relay = Relay()

    print("Attempting to initialize relay...")
    success = relay.initiate_relay()

    if success:
        print("✅ Relay initialized successfully!")

        # Test basic operations
        try:
            print("\nTesting relay state query...")
            status = relay.state(0)  # Get all relay states
            print(f"Current relay states: {status}")

            print("\nTesting individual relay control...")
            # Test relay 1
            relay.state(1, True)   # Turn on
            print("Relay 1: ON")
            import time
            time.sleep(1)
            relay.state(1, False)  # Turn off
            print("Relay 1: OFF")

        except Exception as e:
            print(f"⚠️ Relay operations failed: {e}")
        finally:
            relay.cleanup()

    else:
        print("❌ Relay initialization failed")
        return False

    return True

if __name__ == "__main__":
    if os.geteuid() == 0:
        print("🔑 Running with sudo privileges")
    else:
        print("⚠️ Not running with sudo - may have permission issues")

    success = test_relay()
    sys.exit(0 if success else 1)
