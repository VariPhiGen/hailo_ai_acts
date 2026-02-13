import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone, activity_active_time
)

class HarnessViolation:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.parameters = parameters
        self.zone_data = zone_data
        
        # Track people we have already sent a violation for (to avoid spamming Kafka)
        self.violation_sent_for = set() 
        
        # Track last time we sent a frame to YOLOE for a specific person
        self.last_checked_time = {} 
        self.check_interval = 1.0  # How often to ask YOLOE to check the person (in seconds)

        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        self.timezone = pytz.timezone(self.timezone_str)

    def run(self):
        if not activity_active_time(self.parameters, self.timezone):
            return

        # Throttle the overall activity loop
        if time.time() - self.parameters["last_check_time"] > 1:
            self.parameters["last_check_time"] = time.time()
            current_time = time.time()

            # 1. Find Persons (Using Hailo)
            person_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in self.parameters.get("subcategory_mapping", ["person"])
            ]

            for idx in person_indices:
                tracker_id = self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]
                box = self.parent.detection_boxes[idx] # [xmin, ymin, xmax, ymax]
                
                # Check Zone
                for zone_name, zone_polygon in self.zone_data.items():
                    if is_bottom_in_zone(anchor, zone_polygon):
                        
                        # 2. TRIGGER YOLOE 
                        if hasattr(self.parent, 'yoloe_handler') and self.parent.yoloe_handler:
                            
                            # If we already sent a violation for this person, don't keep checking
                            if tracker_id in self.violation_sent_for:
                                continue

                            # Check cooldown for querying YOLOE about this specific person
                            if tracker_id in self.last_checked_time:
                                if (current_time - self.last_checked_time[tracker_id]) < self.check_interval:
                                    continue
                                    
                            self.last_checked_time[tracker_id] = current_time
                            
                            # Define the Rules for YOLOE
                            meta = {
                                "sensor_id": self.parent.sensor_id,
                                "activity": "HarnessViolation",
                                "hailo_tracker_id": tracker_id,
                                
                                # The Logic Rule: Must find BOTH items
                                "rule": "must_detect",         
                                "required_items": ["harness", "hook"], 
                                "target_bbox": box             
                            }

                            print(f"🚀 YOLOE Check: Person ID {tracker_id} in {zone_name}...", flush=True)

                            self.parent.yoloe_handler.trigger(
                                frame=self.parent.image, 
                                activity_name="HarnessViolation", 
                                metadata=meta
                            )

    def cleaning(self):
        """Clean up tracking data when a person leaves the frame"""
        # If the person is no longer tracked by Hailo, remove them from our memory
        active_trackers = self.parent.last_n_frame_tracker_ids
        
        self.last_checked_time = {
            k: v for k, v in self.last_checked_time.items() 
            if k in active_trackers
        }
        
        self.violation_sent_for = {
            tracker_id for tracker_id in self.violation_sent_for 
            if tracker_id in active_trackers
        }
