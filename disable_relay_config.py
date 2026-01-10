#!/usr/bin/env python3
"""
Script to disable relay functionality in configuration.json
"""

import json
import os

def disable_relay_features():
    config_file = "configuration.json"

    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return False

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Disable relay for all activities
        activities_data = config.get('activities_data', {})

        relay_disabled = []
        for activity_name, activity_config in activities_data.items():
            if 'relay' in activity_config.get('parameters', {}):
                activity_config['parameters']['relay'] = 0
                relay_disabled.append(activity_name)

        # Save the modified configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        if relay_disabled:
            print(f"✅ Disabled relay for activities: {', '.join(relay_disabled)}")
            print("   Activities will still detect violations but won't trigger relays")
        else:
            print("⚠️ No relay-enabled activities found in configuration")

        return True

    except Exception as e:
        print(f"❌ Error modifying configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Disabling Relay Features")
    print("=" * 30)
    success = disable_relay_features()
    if success:
        print("\n✅ Configuration updated! Run detection without relay issues:")
        print("   python3 basic_pipelines/detection.py --i /path/to/video.mp4")
