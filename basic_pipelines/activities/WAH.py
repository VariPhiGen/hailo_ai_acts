import time
import copy
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone,
    is_left_in_zone,
    is_right_in_zone,
    is_object_in_zone,
    xywh_original_percentage
)
from shapely.geometry import Point, Polygon

class WAH:
    def __init__(self, parent,zone_data,parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent=parent

        self.parameters = parameters
        self.zone_data = zone_data
        self.running_data = {}
        self.safe_until = {}

        # Default relay attributes
        self.relay = None
        self.switch_relay = []

        # Pre-evaluate zone check setting (only checked once at startup)
        self.zone_check_enabled = self.parameters.get("no_zone", 0) != 1

        #Initialize Relay
        if parameters["relay"]==1:
            try:
                if self.parent.relay_handler.device==None:
                    success = self.parent.relay_handler.initiate_relay()
                    if not success:
                        print("⚠️ Relay device not available. Continuing without relay control.")
                        self.relay = None
                else:
                    self.relay=self.parent.relay_handler
                    self.switch_relay=parameters["switch_relay"]
            except Exception as e:
                print(f"⚠️ Relay initialization failed: {e}. Continuing without relay control.")
                self.relay = None
        else:
            self.relay = None

        # Initiating Zone wise
        for zone_name in zone_data.keys():
            self.running_data[zone_name]={}

        # Initiating Acts wise
        self.violation_id_data = {}
        for acts in parameters["subcategory_mapping"]:
            self.violation_id_data[acts]=[]
        # Track missing WAH PPE violations
        self.violation_id_data["missing"] = []
        
        self.last_check_time = self.parameters.get("last_check_time", 0)
        self.timezone_str = self.parameters.get("timezone") or self.parent.timezone_str
        self.timezone = pytz.timezone(self.timezone_str)

    def run(self):
        if time.time()-self.parameters["last_check_time"]>2:
            self.parameters["last_check_time"]=time.time()
            wah_objects=self.parameters["subcategory_mapping"]
            
            # Snapshot the current frame once for this entire run() cycle.
            # self.parent.image is a live reference overwritten every frame by the GStreamer
            # callback; capturing a copy here guarantees the violation image matches the
            # detections we are about to analyse (no ghost frames, no off-by-one frame).
            current_frame = self.parent.image.copy() if self.parent.image is not None else None
            if current_frame is None:
                return
            
            # Get condition_label from parameters (replaces check_zone_label)
            condition_label = self.parameters.get("condition_label", None)
            # If condition_label is empty/None, bypass condition check entirely
            condition_label_enabled = condition_label is not None and condition_label != "" and (
                (isinstance(condition_label, list) and len(condition_label) > 0) or
                (not isinstance(condition_label, list) and condition_label != "")
            )
            
            # Normalize condition_label to list
            if condition_label_enabled:
                if isinstance(condition_label, list):
                    condition_labels = condition_label
                else:
                    condition_labels = [condition_label]
            else:
                condition_labels = []
            
            # Get YOLOE results for condition_label segmentation
            yoloe_result_data = None
            yoloe_confidence_threshold = self.parameters.get("yoloe_confidence", 0.0)
            # Max age (seconds) before a YOLOE result is considered too stale to use.
            # Configurable via parameters["yoloe_max_stale_age"] in configuration.json.
            yoloe_max_stale_age = self.parameters.get("yoloe_max_stale_age", 60)
            
            if condition_label_enabled:
                with self.parent.yoloe_lock:
                    yoloe_results = self.parent.yoloe_results
                    if yoloe_results and yoloe_results.get("result"):
                        # Staleness guard: discard results older than yoloe_max_stale_age seconds
                        last_run = yoloe_results.get("last_run", 0.0)
                        if (time.time() - last_run) <= yoloe_max_stale_age:
                            # Deep-copy inside the lock to prevent concurrent mutation by
                            # the YOLOE background thread overwriting the dict mid-iteration.
                            yoloe_result_data = copy.deepcopy(yoloe_results["result"])
            
            person_indices = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
            wah_indices = [i for i, cls in enumerate(self.parent.classes) if cls in wah_objects.keys()]

            for idx in person_indices:
                box=self.parent.detection_boxes[idx]
                obj_class=self.parent.classes[idx]
                tracker_id=self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]
                person_detection_score = self.parent.detection_score[idx]
                if person_detection_score < 0.7:
                    continue
                
                person_poly = Polygon([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]),  (box[2], box[1])])
                
                # Check if person is inside any detected condition_label segmentation polygon from YOLOE
                # Only if condition_label is configured
                person_inside_condition_zone = True  # Default to True if condition_label is disabled
                if condition_label_enabled:
                    # If YOLOE results are None or empty, skip WAH checks for this person
                    if yoloe_result_data is None:
                        continue
                    
                    person_inside_condition_zone = False
                    yoloe_prompts = yoloe_result_data.get("prompt", [])
                    yoloe_polygons = yoloe_result_data.get("polygon", [])
                    yoloe_confidences = yoloe_result_data.get("confidence", [])
                    
                    # Match condition_labels with YOLOE results and check if person is inside any polygon
                    for i, detected_label in enumerate(yoloe_prompts):
                        # Check if this detected label matches any of our condition_labels
                        if detected_label not in condition_labels:
                            continue
                        
                        # Check confidence threshold if configured
                        if i < len(yoloe_confidences) and yoloe_confidence_threshold > 0:
                            if yoloe_confidences[i] < yoloe_confidence_threshold:
                                continue
                        
                        # Get polygon for this detection
                        if i < len(yoloe_polygons) and yoloe_polygons[i]:
                            try:
                                # Convert polygon from [[x1,y1], [x2,y2], ...] to Shapely Polygon
                                polygon_points = [(float(pt[0]), float(pt[1])) for pt in yoloe_polygons[i]]
                                if len(polygon_points) >= 3:  # Need at least 3 points for a valid polygon
                                    condition_poly = Polygon(polygon_points)
                                    # Check if person is inside or intersects the condition_label polygon
                                    if person_poly.intersects(condition_poly) or condition_poly.contains(person_poly):
                                        person_inside_condition_zone = True
                                        break
                            except Exception as e:
                                print(f"DEBUG: Error processing YOLOE polygon for {detected_label}: {e}")
                                continue

                # Only proceed with WAH violation checks if person is inside condition_label (or if check is disabled)
                if not person_inside_condition_zone:
                    continue
                
                # Determine which zones to process
                if not self.zone_check_enabled:
                    # Zones disabled - use default zone
                    zones_to_process = ["default"]
                    if "default" not in self.running_data:
                        self.running_data["default"] = {}
                else:
                    # Zones enabled - find zones where person is located
                    zones_to_process = []
                    for zone_name, zone_polygon in self.zone_data.items():
                        # Check if person is in this zone
                        if is_bottom_in_zone(anchor, zone_polygon):
                            zones_to_process.append(zone_name)
                    
                    # If person is not in any zone, skip
                    if not zones_to_process:
                        continue
                
                # Process WAH checks for each zone
                for zone_name in zones_to_process:
                    # Check if any required WAH PPE (e.g., harness/hooks) is present
                    wah_found = False
                    for wah_idx in wah_indices:
                        wah_box=self.parent.detection_boxes[wah_idx]
                        wah_obj_class=self.parent.classes[wah_idx]
                        wah_confidence=self.parent.detection_score[wah_idx]
                        if wah_confidence < 0.7:
                            continue
                        if is_left_in_zone(wah_box, person_poly) or is_right_in_zone(wah_box, person_poly):
                            wah_found = True
                            break

                    if tracker_id not in self.running_data[zone_name]:
                        self.running_data[zone_name][tracker_id] = {"missing": 0}

                    if wah_found:
                        # Mark person as safe for 5 minutes if WAH PPE is detected
                        self.safe_until[tracker_id] = time.time() + 300
                        # Reset missing counter when any required PPE is detected
                        self.running_data[zone_name][tracker_id]["missing"] = 0
                        continue

                    # Skip further processing if person is marked safe
                    if time.time() < self.safe_until.get(tracker_id, 0):
                        continue

                    # Increment missing counter when no required PPE is detected
                    self.running_data[zone_name][tracker_id]["missing"] += 1

                    if self.running_data[zone_name][tracker_id]["missing"] > self.parameters["frame_accuracy"]:
                        if self.relay!=None and self.parameters["relay"]==1:
                            try:
                                status=self.relay.state(0)
                                true_indexes = [(i+1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                for index in self.switch_relay:
                                    if (index) not in true_indexes:
                                        self.relay.state(index, on=True)
                                    self.relay.start_time[index]=time.time()
                                    print("Changed the Index")
                            except Exception as e:
                                print(f"⚠️ Relay operation failed: {e}. Continuing without relay control.")

                        if tracker_id not in self.violation_id_data["missing"]:
                            self.violation_id_data["missing"].append(tracker_id)

                            xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)

                            ############################################################
                            now_local = datetime.now(self.timezone)
                            fake_utc = now_local.replace(tzinfo=pytz.utc)
                            datetimestamp = f"{fake_utc.isoformat()}"
                            ############################################################

                            missing_label = self.parameters.get(
                                "missing_subcategory",
                                "No " + " or ".join(wah_objects.values())
                            )

                            # Include zone info only if zones are being enforced
                            event_metadata = {}
                            if self.zone_check_enabled:
                                event_metadata["zone_name"] = zone_name

                            self.parent.create_result_events(
                                xywh,
                                obj_class,
                                f"PPE-{missing_label}",
                                event_metadata,
                                datetimestamp,
                                person_detection_score,
                                current_frame,
                            )

    def cleaning(self):
        # Prune per-act violations to only tracker_ids seen in recent frames
        for acts in list(self.violation_id_data.keys()):
            self.violation_id_data[acts] = [
                tracker_id
                for tracker_id in self.violation_id_data[acts]
                if tracker_id in self.parent.last_n_frame_tracker_ids
            ]

        # Prune safe tracker_ids to only active ones
        self.safe_until = {
            tracker_id: expiry
            for tracker_id, expiry in self.safe_until.items()
            if tracker_id in self.parent.last_n_frame_tracker_ids
        }

        # Prune running_data per zone to only active tracker_ids
        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                key: value
                for key, value in self.running_data[zone_name].items()
                if key in self.parent.last_n_frame_tracker_ids
            }
        
