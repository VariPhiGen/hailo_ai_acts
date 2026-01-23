import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone, xywh_original_percentage,
    init_relay,
    trigger_relay,
    relay_auto_off,
    activity_active_time
)

class UnauthorisedArea:
    def __init__(self, parent, zone_data, parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent = parent
        self.parameters = parameters
        self.zone_data = zone_data
        self.person_entry_times = {}  # Track when each person entered the zone
        self.violation_id_data = []

        # Initialize Relay
        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        self.timezone = pytz.timezone(self.timezone_str)

    def run(self):
        if not activity_active_time(self.parameters, self.timezone):
            print("[UnauthorisedArea] Out of active time")
            return
        
        # print("[UnauthorisedArea] called")
        """Main entry point for this activity"""
        if time.time()-self.parameters["last_check_time"]>1:
            self.parameters["last_check_time"]=time.time()
            
            current_time = time.time()

            # Get indices of people/objects we want to track
            offender_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in self.parameters["subcategory_mapping"]
            ]

            for idx in offender_indices:
                obj_class = self.parent.classes[idx]
                tracker_id = self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]

                for zone_name, zone_polygon in self.zone_data.items():
                    zone_tracker_key = f"{zone_name}_{tracker_id}"

                    if is_bottom_in_zone(anchor, zone_polygon):
                        # Person inside unauthorized zone
                        print("person found in zone", flush=True)

                        if zone_tracker_key not in self.person_entry_times:
                            # Entry time
                            self.person_entry_times[zone_tracker_key] = current_time
                            continue

                        time_in_zone = current_time - self.person_entry_times[zone_tracker_key]

                        if (
                            time_in_zone > self.parameters["time_limit"]
                            and tracker_id not in self.violation_id_data
                        ):
                            self.violation_id_data.append(tracker_id)
                            print("Violation found: {tracker_id}", flush=True)

                            # Trigger relay if configured
                            if self.parameters["relay"] == 1:
                                trigger_relay(self.relay, self.switch_relay)

                            # Create violation event
                            box = self.parent.detection_boxes[idx]
                            xywh = xywh_original_percentage(box)
                            datetimestamp = f"{datetime.now(self.timezone).isoformat()}"

                            self.parent.create_result_events(
                                xywh,
                                obj_class,
                                "Security-Unauthorized Area",
                                {"zone_name": zone_name},
                                datetimestamp,
                                1,
                                self.parent.image
                            )
                    else:
                        # Person left the zone → reset dwell timer
                        if zone_tracker_key in self.person_entry_times:
                            del self.person_entry_times[zone_tracker_key]

    def cleaning(self):
        """Clean up tracking data for persons that are no longer detected"""
        # Remove violation IDs for persons no longer in frame
        self.violation_id_data = [tracker_id for tracker_id in self.violation_id_data
                                if tracker_id in self.parent.last_n_frame_tracker_ids]

        # Remove entry times for persons no longer in frame
        keys_to_remove = []
        for zone_tracker_key in self.person_entry_times.keys():
            zone_name, tracker_id = zone_tracker_key.rsplit('_', 1)
            if tracker_id not in self.parent.last_n_frame_tracker_ids:
                keys_to_remove.append(zone_tracker_key)

        for key in keys_to_remove:
            del self.person_entry_times[key]
