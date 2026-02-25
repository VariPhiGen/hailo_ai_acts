import time
from datetime import datetime
from shapely.geometry import Point

# ANSI Color codes for terminal printing
RED = "\033[91m"
RESET = "\033[0m"

from activities.activity_helper_utils import (
    init_relay,
    trigger_relay,
    relay_auto_off,  # <--- Make sure this is imported
    xywh_original_percentage,
    activity_active_time,
    calculate_iou
)

class UnauthorisedActivity:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters

        # Per-zone tracker dwell state: {zone_name: {tracker_id: start_time}}
        self.running_data = {zone: {} for zone in zone_data.keys()}

        # Dedup: list of IDs that have already triggered to prevent spamming
        self.violation_id_data = []

        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

    def run(self):
        # 1. Manage Relay Auto-off (Crucial: checks every frame if relay should turn off)
        if self.parameters["relay"] == 1:
            relay_auto_off(self.relay, self.switch_relay)

        # 2. Check Schedule
        if not activity_active_time(self.parameters, self.parent.ist_timezone):
            return

        # 3. Rate limit (runs logic once per second)
        if time.time() - self.parameters.get("last_check_time", 0) <= 1:
            return
        self.parameters["last_check_time"] = time.time()

        dwell_time_sec = self.parameters["dwell_time_sec"]

        # No pose results → skip
        if not self.parent.pose_detection_boxes:
            return

        # Loop over pose persons
        for pose_box, keypoints in zip(self.parent.pose_detection_boxes, self.parent.pose_keypoints):

            # Step 1: Match pose box to detection box → get tracker ID
            tracker_id = None
            for i, det_box in enumerate(self.parent.detection_boxes):
                iou = calculate_iou(self, pose_box, det_box) # Passed self if calculate_iou needs it, or remove self if static
                if iou > 0.5:
                    try:
                        tracker_id = self.parent.tracker_ids[i]
                        break
                    except IndexError:
                        continue

            if tracker_id is None:
                continue

            # Step 2: Check each forbidden zone
            for zone_name, zone_polygon in self.zone_data.items():
                
                # Check if ANY valid keypoint is inside the zone
                touched = False
                for kp_x, kp_y, conf in keypoints:
                    if conf < 0.3: 
                        continue
                    if zone_polygon.contains(Point(kp_x, kp_y)):
                        touched = True
                        break

                zone_state = self.running_data[zone_name]

                if touched:
                    # Person is INSIDE zone
                    if tracker_id not in zone_state:
                        # First time seeing them in zone -> Start Timer
                        zone_state[tracker_id] = time.time()
                    else:
                        # They are still in zone -> Check Dwell Time
                        elapsed = time.time() - zone_state[tracker_id]

                        if elapsed >= dwell_time_sec and tracker_id not in self.violation_id_data:
                            # --- VIOLATION CONFIRMED ---
                            
                            # 1. Add to Ignore list (so we don't spam 100 alerts per second)
                            self.violation_id_data.append(tracker_id)

                            # 2. Trigger Hardware
                            if self.parameters["relay"] == 1:
                                trigger_relay(self.relay, self.switch_relay)

                            # 3. Prepare Data
                            xywh = xywh_original_percentage(
                                pose_box,
                                self.parent.original_width,
                                self.parent.original_height
                            )
                            datetimestamp = f"{datetime.now(self.parent.ist_timezone).isoformat()}_{tracker_id}"

                            # 4. PRINT TO TERMINAL (High Visibility)
                            print(f"{RED} [ALARM] UNAUTHORIZED ENTRY DETECTED!{RESET}")
                            print(f"{RED} └── ID: {tracker_id} | ZONE: {zone_name} | TIME: {elapsed:.1f}s{RESET}")

                            # 5. Create Event for Dashboard/DB
                            self.parent.create_result_events(
                                xywh,
                                "person",
                                "Unauthorised_Touch",
                                {"zone_name": zone_name},
                                datetimestamp,
                                1,
                                self.parent.image
                            )
                else:
                    # Person is OUTSIDE zone
                    
                    # Reset Dwell Timer
                    if tracker_id in zone_state:
                        del zone_state[tracker_id]
                    
                    # Reset Violation State (Fixing the latch bug)
                    # If they leave the zone, we allow them to be detected again if they return.
                    if tracker_id in self.violation_id_data:
                         self.violation_id_data.remove(tracker_id)
                         print(f" [INFO] ID {tracker_id} left zone {zone_name}. Resetting violation status.")

    def cleaning(self):
        # Remove data for IDs that are no longer tracked by the camera
        current_ids = self.parent.last_n_frame_tracker_ids
        
        # Clean violation list
        self.violation_id_data = [
            tid for tid in self.violation_id_data
            if tid in current_ids
        ]

        # Clean running timers
        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                tid: start_time
                for tid, start_time in self.running_data[zone_name].items()
                if tid in current_ids
            }
