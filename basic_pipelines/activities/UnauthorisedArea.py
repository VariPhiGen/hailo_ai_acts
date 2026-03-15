import time
from datetime import datetime
import pytz
from shapely.geometry import Point, Polygon
from activities.activity_helper_utils import (
    is_bottom_in_zone, xywh_original_percentage,
    init_relay, trigger_relay, activity_active_time
)

class UnauthorisedArea:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.parameters = parameters
        self.zone_data = zone_data
        self.person_entry_times = {}  
        self.violation_id_data = []

        # Initialize Relay
        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)
        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        self.timezone = pytz.timezone(self.timezone_str)

    def run(self):
        if not activity_active_time(self.parameters, self.timezone):
            return
        
        if time.time() - self.parameters["last_check_time"] > 1:
            self.parameters["last_check_time"] = time.time()
            current_time = time.time()

            # --- 1. Fetch YOLOE Condition Settings ---
            condition_label = self.parameters.get("condition_label", None)
            condition_label_enabled = condition_label is not None and condition_label != "" and (
                (isinstance(condition_label, list) and len(condition_label) > 0) or
                (not isinstance(condition_label, list) and condition_label != "")
            )
            
            condition_labels = condition_label if isinstance(condition_label, list) else [condition_label] if condition_label_enabled else []
            yoloe_confidence_threshold = self.parameters.get("yoloe_confidence", 0.1)
            
            # Safely get YOLOE results from the central dictionary
            yoloe_result_data = None
            if condition_label_enabled:
                with self.parent.yoloe_lock:
                    yoloe_results = self.parent.yoloe_results
                    if yoloe_results and yoloe_results.get("result"):
                        yoloe_result_data = yoloe_results["result"]

            # --- 2. Evaluate Hailo Detections against YOLOE Polygons ---
            offender_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in self.parameters["subcategory_mapping"]
            ]

            for idx in offender_indices:
                obj_class = self.parent.classes[idx]
                tracker_id = self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]
                box = self.parent.detection_boxes[idx]
                
                # Create a Shapely Polygon for the person bounding box
                person_poly = Polygon([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])])
                
                # Check condition logic (intersect with YOLOE mask)
                person_inside_condition_zone = True  # Default to True if YOLOE isn't active
                if condition_label_enabled:
                    if yoloe_result_data is None:
                        continue # Wait until YOLOE data is available
                    
                    person_inside_condition_zone = False
                    yoloe_prompts = yoloe_result_data.get("detections", [])
                    
                    for det in yoloe_prompts:
                        detected_label = det.get("prompt")
                        if detected_label not in condition_labels:
                            continue
                        
                        if det.get("confidence", 0) < yoloe_confidence_threshold:
                            continue
                            
                        # Extract Polygon
                        polygon_coords = det.get("polygon", [])
                        if polygon_coords:
                            try:
                                polygon_points = [(float(pt[0]), float(pt[1])) for pt in polygon_coords]
                                if len(polygon_points) >= 3:
                                    condition_poly = Polygon(polygon_points)
                                    # If the person is touching the scaffolding mask
                                    if person_poly.intersects(condition_poly) or condition_poly.contains(person_poly):
                                        person_inside_condition_zone = True
                                        break
                            except Exception as e:
                                print(f"DEBUG: Error processing YOLOE polygon: {e}")
                                continue

                # If YOLOE condition is enabled but they aren't on the scaffolding, ignore them
                if not person_inside_condition_zone:
                    continue

                # --- 3. Run Standard Zone Entry/Dwell Time Logic ---
                for zone_name, zone_polygon in self.zone_data.items():
                    zone_tracker_key = f"{zone_name}_{tracker_id}"

                    if is_bottom_in_zone(anchor, zone_polygon):
                        if zone_tracker_key not in self.person_entry_times:
                            self.person_entry_times[zone_tracker_key] = current_time
                            continue

                        time_in_zone = current_time - self.person_entry_times[zone_tracker_key]

                        if (time_in_zone > self.parameters["time_limit"] 
                            and tracker_id not in self.violation_id_data):
                            
                            self.violation_id_data.append(tracker_id)
                            print(f"Violation found: Person {tracker_id} stayed on {condition_labels[0]} too long!", flush=True)

                            if self.parameters["relay"] == 1:
                                trigger_relay(self.relay, self.switch_relay)

                            xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                            datetimestamp = f"{datetime.now(self.timezone).isoformat()}"

                            self.parent.create_result_events(
                                xywh, obj_class, "Unauthorized Area", 
                                {"zone_name": zone_name, "condition": condition_labels[0]}, 
                                datetimestamp, 1, self.parent.image
                            )
                    else:
                        if zone_tracker_key in self.person_entry_times:
                            del self.person_entry_times[zone_tracker_key]

    def cleaning(self):
        self.violation_id_data = [tracker_id for tracker_id in self.violation_id_data
                                if tracker_id in self.parent.last_n_frame_tracker_ids]

        keys_to_remove = []
        for zone_tracker_key in self.person_entry_times.keys():
            zone_name, tracker_id = zone_tracker_key.rsplit('_', 1)
            if tracker_id not in self.parent.last_n_frame_tracker_ids:
                keys_to_remove.append(zone_tracker_key)

        for key in keys_to_remove:
            del self.person_entry_times[key]
