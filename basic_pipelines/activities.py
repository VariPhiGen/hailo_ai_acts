import time
from datetime import datetime
import pytz
from shapely.geometry import Polygon
import numpy as np
from collections import deque
import cv2
import numpy as np
import math
from collections import defaultdict

from shapely.geometry import Polygon, box as shapely_box

from activity_helper_utils import (
    is_bottom_in_zone,
    is_object_in_zone,
    xywh_original_percentage,
    activity_active_time,
    init_relay,
    trigger_relay,
    relay_auto_off,
    calculate_iou,
    extract_median_color,
    uniform_validation,
    bottom_center,
    get_interpolated_ppm,
    merge_box,
    get_zone_grayscale,
    compute_histogram
)


class AnalyticsEngine:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config

        # ── Relay storage (mirrors raw file: self.relay, self.switch_relay) ──
        self.relays = {}
        self.switch_relays = {}

        # ── Timezone storage (one per activity, from params["timezone"]) ──
        self.timezones = {}

        # ── State storage ──
        self.active_methods = []
        self.zone_data = {}
        self.parameters_data = {}
        self.running_data = {}
        self.violation_id_data = {}

        # ── Config keys ──
        available_activities = config.get("available_activities", [])
        active_activities    = config.get("active_activities", [])
        activities_data      = config.get("activities_data", {})

        for activity, details in activities_data.items():

            # Skip if not available or not active
            if activity not in available_activities or activity not in active_activities:
                continue

            params = details.get("parameters", {})

            # Timezone — read from params["timezone"] exactly like raw file
            tz_str = params.get("timezone", "Asia/Kolkata")
            self.timezones[activity] = pytz.timezone(tz_str)

            # last_check_time — reset to now so the 1-second gate works immediately
            params["last_check_time"] = time.time()

            # Relay — mirrors raw file: init_relay(self.parent, self.parameters)
            relay_obj, switch_relay = init_relay(self.parent, params)
            self.relays[activity]       = relay_obj
            self.switch_relays[activity] = switch_relay

            self.parameters_data[activity] = params

            # Zones — wrap coords in Polygon, same as raw file receives
            if "zones" in details and activity != "queue_analytics":
                self.zone_data[activity] = {
                    zone: Polygon(coords)
                    for zone, coords in details["zones"].items()
                }

            # ── Activity-specific state init ──
            if activity == "UnauthorisedArea":
                self.running_data["UnauthorisedArea"] = {"person_entry_times": {}}
                self.violation_id_data["UnauthorisedArea"] = []
                self.active_methods.append(self.unauthorised_area)
                
            elif activity == "PeopleGathering":
                self.running_data["PeopleGathering"] = {}   # zone_name -> frame_count
                self.violation_id_data["PeopleGathering"] = []
                self.active_methods.append(self.people_gathering)
                
            elif activity == "UnsafeZone":
                self.running_data["UnsafeZone"] = {zone: {} for zone in self.zone_data["UnsafeZone"].keys()}
                self.violation_id_data["UnsafeZone"] = []
                self.active_methods.append(self.unsafe_zone)
            
            elif activity == "PPE":
                self.running_data["PPE"] = {zone: {} for zone in self.zone_data["PPE"].keys()}
                self.violation_id_data["PPE"] = {
                    acts: [] for acts in params.get("subcategory_mapping", {}).keys()
                }
                self.ppe_zone_check_enabled = params.get("no_zone", 0) != 1
                self.active_methods.append(self.ppe)
                
            elif activity == "TimeBasedUnauthorizedAccess":
                self.running_data["TimeBasedUnauthorizedAccess"] = {
                    zone: {} for zone in self.zone_data["TimeBasedUnauthorizedAccess"].keys()}
                self.violation_id_data["TimeBasedUnauthorizedAccess"] = []
                self.active_methods.append(self.time_based_unauthorized_access)
                
            elif activity == "StrayParking":
                self.running_data["StrayParking"] = {
                    zone: {} for zone in self.zone_data["StrayParking"].keys()
                }
                self.violation_id_data["StrayParking"] = []
                self.active_methods.append(self.stray_parking)
                
            elif activity == "UnauthorisedVehicleArea":
                self.violation_id_data["UnauthorisedVehicleArea"] = []
                self.active_methods.append(self.unauthorised_vehicle_area)
                
            elif activity == "VehicleCongestion":
                self.running_data["VehicleCongestion"] = {
                    zone: {"start_time": None, "triggered": False}
                    for zone in self.zone_data["VehicleCongestion"].keys()
                }
                self.active_methods.append(self.vehicle_congestion)
                
            elif activity == "MinMaxWorkerCount":
                self.running_data["MinMaxWorkerCount"] = {
                    zone: {
                        "active": False,
                        "violation_type": None,
                        "start_time": None,
                        "start_frame": None,
                        "logged": False
                    }
                    for zone in self.zone_data["MinMaxWorkerCount"].keys()
                }
                self.active_methods.append(self.min_max_worker_count)
                
            elif activity == "WrongLane":
                self.running_data["WrongLane"] = {
                    zone: {} for zone in self.zone_data["WrongLane"].keys()
                }
                self.violation_id_data["WrongLane"] = []
                self.active_methods.append(self.wrong_lane)
                
            elif activity == "unattended_bag_detection":
                self.running_data["unattended_bag_detection"] = {}   # zone_name -> {tracker_id -> {last_attended}}
                self.violation_id_data["unattended_bag_detection"] = []
                self.active_methods.append(self.unattended_bag_detection)
            
            elif activity == "blur_detection":
                self.running_data["blur_detection"] = {}   # blur_key -> last_alert_time
                self.active_methods.append(self.blur_detection)
            
            elif activity == "first_person_entering":
                self.running_data["first_person_entering"] = {}   # zone_name -> last_alert_date
                self.active_methods.append(self.first_person_entering)
                
            elif activity == "last_person_leaving":
                self.running_data["last_person_leaving"] = {}   # zone_name -> zone_state dict
                self.active_methods.append(self.last_person_leaving)
            
            elif activity == "hairnet_detection":
                self.running_data["hairnet_detection"] = {}   # tracker_id -> frame_count
                self.violation_id_data["hairnet_detection"] = []
                self.active_methods.append(self.hairnet_detection)

            elif activity == "broomstick_detection":
                self.running_data["broomstick_detection"] = {}   # tracker_id -> frame_count
                self.violation_id_data["broomstick_detection"] = []
                self.active_methods.append(self.broomstick_detection)
                
            elif activity == "unattended_student_detection":
                self.running_data["unattended_student_detection"] = {}   # zone_name -> {tracker_id -> {last_attended}}
                self.violation_id_data["unattended_student_detection"] = []
                self.active_methods.append(self.unattended_student_detection)
                
            elif activity == "Heatmap_overlay":
                self.running_data["Heatmap_overlay"] = {
                    "heatmap_matrix": None,
                    "heatmap_image_to_send": None,
                    "window_active_previously": False
                }
                self.active_methods.append(self.heatmap_overlay)
                
            elif activity == "queue_analytics":
                self.running_data["queue_analytics"] = {
                    "queue_data":        defaultdict(dict),
                    "queue_confirmed":   defaultdict(set),
                    "queue_exit_time":   defaultdict(dict),
                    "counter_data":      defaultdict(dict),
                    "counter_confirmed": defaultdict(set),
                    "counter_visited":   defaultdict(set),
                    "service_history":   defaultdict(list),
                    "abandonment_count": defaultdict(int),
                }
                # Special zone init — queue_area and counter_area per zone
                self.zone_data["queue_analytics"] = {}
                for zone_name, zone_config in details.get("zones", {}).items():
                    self.zone_data["queue_analytics"][zone_name] = {
                        "queue_area":   Polygon(zone_config["queue_area"]),
                        "counter_area": Polygon(zone_config["counter_area"])
                    }
                self.active_methods.append(self.queue_analytics)
                
            elif activity == "Loitering":
                # running_data = loitering_data (per zone per tracker counter)
                self.running_data["Loitering"] = {
                    zone: {} for zone in self.zone_data["Loitering"].keys()
                }
                self.violation_id_data["Loitering"] = []
                # hsv targets and tolerances stored once at init from params
                self.loitering_hsv_target    = params.get("hsv_target", {})
                self.loitering_hsv_tolerance = params.get("hsv_tolerance", {})
                self.active_methods.append(self.loitering)
                
            elif activity == "UnauthorisedAccess":
                self.running_data["UnauthorisedAccess"] = {
                    zone: {} for zone in self.zone_data["UnauthorisedAccess"].keys()
                }
                self.violation_id_data["UnauthorisedAccess"] = []
                self.active_methods.append(self.unauthorised_access)
            
            elif activity == "IdlePerson":
                self.running_data["IdlePerson"] = {
                    zone: {} for zone in self.zone_data["IdlePerson"].keys()
                }
                self.violation_id_data["IdlePerson"] = []
                self.active_methods.append(self.idle_person)
                
            elif activity == "RunningDetection":
                # violation_id_data is a DICT keyed by tracker_id (not a list)
                self.running_data["RunningDetection"] = {}   # tracker_id: {positions, timestamps}
                self.violation_id_data["RunningDetection"] = {}  # tracker_id: frame_number
                self.active_methods.append(self.running_detection)
                
            elif activity == "VehicleInteraction":
                self.running_data["VehicleInteraction"] = {
                    "prev_pos":       {},   # tracker_id: box
                    "prev_distances": {}    # (id1, id2) tuple: distance
                }
                self.violation_id_data["VehicleInteraction"] = []
                self.active_methods.append(self.vehicle_interaction)
                
            elif activity == "LightDetection":
                # violation_id_data is a DICT keyed by zone_name (not a list)
                # each zone holds: light_state, state_change_frame, violation_start_frame
                self.violation_id_data["LightDetection"] = {}

                # calibration state stored in running_data
                self.running_data["LightDetection"] = {
                    "calibration_done":  False,
                    "calibration_state": None,   # populated during calibrate step
                    "calibration_data":  {       # holds averaged ON/OFF histograms
                        "ON":  None,
                        "OFF": None
                    }
                }
                self.active_methods.append(self.light_detection)
                
            elif activity == "EntryExitWLELogs":
                # special zone structure: each zone has TWO polygons [entry_polygon, exit_polygon]
                # stored separately from self.zone_data since structure is different
                self.entry_exit_zones = {
                    zone: [Polygon(coords[0]), Polygon(coords[1])]
                    for zone, coords in details["zones"].items()
                }
                # per zone, per class tracking of ids and counts
                self.running_data["EntryExitWLELogs"] = {}
                for zone_name in details["zones"].keys():
                    self.running_data["EntryExitWLELogs"][zone_name] = {
                        "entry_ids":   {cls: [] for cls in params.get("subcategory_mapping", [])},
                        "exit_ids":    {cls: [] for cls in params.get("subcategory_mapping", [])},
                        "entry_count": {cls: 0  for cls in params.get("subcategory_mapping", [])},
                        "exit_count":  {cls: 0  for cls in params.get("subcategory_mapping", [])}
                    }
                # lc_data mirrors raw self.lc_data — running totals reset after each send
                self.running_data["EntryExitWLELogs"]["lc_data"] = {
                    "Entry": 0,
                    "Exit":  0
                }
                self.active_methods.append(self.entry_exit_wle_logs)
                
            elif activity == "WorkforceEfficiency":
                # per zone person count + frame counter + timestamp
                self.running_data["WorkforceEfficiency"] = {
                    zone: 0 for zone in self.zone_data["WorkforceEfficiency"].keys()
                }
                self.running_data["WorkforceEfficiency"]["current_frame_count"] = 0
                self.running_data["WorkforceEfficiency"]["send_data_frame_limit"] = params.get("send_data_frame_limit", 300)
                self.running_data["WorkforceEfficiency"]["timestamp"] = datetime.now(
                    self.timezones["WorkforceEfficiency"]
                ).isoformat()
                self.running_data["WorkforceEfficiency"]["time"] = time.time()
                self.active_methods.append(self.workforce_efficiency)
                
            elif activity == "UnattendedArea":
                self.running_data["UnattendedArea"] = {
                    zone: {
                        "last_security_guard_seen": time.time(),  # assume guard present at start
                        "alert_sent":    False,
                        "cooldown_until": 0.0
                    }
                    for zone in self.zone_data["UnattendedArea"].keys()
                }
                self.active_methods.append(self.unattended_area)
                
            # yoloe test for text and visual and both
            elif activity == "YoloeTest":
                self.running_data["YoloeTest"] = {
                    "person_entry_times": {}
                }
                self.violation_id_data["YoloeTest"] = []
                self.active_methods.append(self.yoloe_test)
                print("✅ YoloeTest registered.")
                
                
# ---------------------- Activities fucntions ---------------------------------------
    # ─────────────────────────────────────────────────────────────────────────
    # UnauthorizedArea
    # ─────────────────────────────────────────────────────────────────────────
    def unauthorised_area(self):
        params       = self.parameters_data["UnauthorisedArea"]
        tz           = self.timezones["UnauthorisedArea"]
        relay        = self.relays["UnauthorisedArea"]
        switch_relay = self.switch_relays["UnauthorisedArea"]
        
        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            current_time  = time.time()
            entry_times   = self.running_data["UnauthorisedArea"]["person_entry_times"]
            violation_ids = self.violation_id_data["UnauthorisedArea"]

            offender_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in params["subcategory_mapping"]
            ]

            for idx in offender_indices:
                obj_class  = self.parent.classes[idx]
                tracker_id = self.parent.tracker_ids[idx]
                anchor     = self.parent.anchor_points_original[idx]

                for zone_name, zone_polygon in self.zone_data["UnauthorisedArea"].items():
                    zone_tracker_key = f"{zone_name}_{tracker_id}"

                    if is_bottom_in_zone(anchor, zone_polygon):

                        if zone_tracker_key not in entry_times:
                            entry_times[zone_tracker_key] = current_time
                            continue

                        time_in_zone = current_time - entry_times[zone_tracker_key]
                        
                        if (
                            time_in_zone > params["time_limit"]
                            and tracker_id not in violation_ids
                        ):
                            violation_ids.append(tracker_id)
                            print("Violation found: ", tracker_id)
                            
                            if params["relay"] == 1:
                                trigger_relay(relay, switch_relay)

                            box  = self.parent.detection_boxes[idx]
                            xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                            datetimestamp = f"{datetime.now(tz).isoformat()}"

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
                        if zone_tracker_key in entry_times:
                            del entry_times[zone_tracker_key]
                            
    # ─────────────────────────────────────────────────────────────────────────
    # people gathering
    # ─────────────────────────────────────────────────────────────────────────                            
    def people_gathering(self):
        params       = self.parameters_data["PeopleGathering"]
        tz           = self.timezones["PeopleGathering"]
        relay        = self.relays["PeopleGathering"]
        switch_relay = self.switch_relays["PeopleGathering"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            frame_counts  = self.running_data["PeopleGathering"]
            violation_ids = self.violation_id_data["PeopleGathering"]

            person_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls == "person"
            ]

            for zone_name, zone_polygon in self.zone_data["PeopleGathering"].items():
                count = sum(
                    1 for idx in person_indices
                    if is_bottom_in_zone(self.parent.anchor_points_original[idx], zone_polygon)
                )

                if count >= params["max_people"]:
                    frame_counts[zone_name] = frame_counts.get(zone_name, 0) + 1
                else:
                    frame_counts[zone_name] = 0

                if (
                    frame_counts[zone_name] >= params["frame_accuracy"]
                    and zone_name not in violation_ids
                ):
                    violation_ids.append(zone_name)

                    if params["relay"] == 1:
                        trigger_relay(relay, switch_relay)

                    datetimestamp = datetime.now(tz).isoformat()
                    print("violation PeopleGathering: ", count)
                    self.parent.create_result_events(
                        None,
                        "person",
                        "People_Gathering",
                        {"zone_name": zone_name, "count": count},
                        datetimestamp,
                        1,
                        self.parent.image
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # unsafezone
    # ─────────────────────────────────────────────────────────────────────────
    def unsafe_zone(self):
        params       = self.parameters_data["UnsafeZone"]
        tz           = self.timezones["UnsafeZone"]
        relay        = self.relays["UnsafeZone"]
        switch_relay = self.switch_relays["UnsafeZone"]

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            # guarded since relay is None when relay==0 in config
            if relay is not None:
                relay.check_auto_off(switch_relay)

            offender_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in params["subcategory_mapping"]
            ]
            print(offender_indices, self.running_data["UnsafeZone"])

            for idx in offender_indices:
                box        = self.parent.detection_boxes[idx]
                obj_class  = self.parent.classes[idx]
                tracker_id = self.parent.tracker_ids[idx]
                anchor     = self.parent.anchor_points_original[idx]
                violation_ids = self.violation_id_data["UnsafeZone"]

                for zone_name, zone_polygon in self.zone_data["UnsafeZone"].items():
                    if is_bottom_in_zone(anchor, zone_polygon) and (
                        tracker_id not in violation_ids or params["relay"] == 1
                    ):
                        if tracker_id not in self.running_data["UnsafeZone"][zone_name]:
                            self.running_data["UnsafeZone"][zone_name][tracker_id] = 1
                        else:
                            self.running_data["UnsafeZone"][zone_name][tracker_id] += 1

                        if self.running_data["UnsafeZone"][zone_name][tracker_id] > params["frame_accuracy"]:

                            if relay is not None and params["relay"] == 1:
                                try:
                                    status = relay.state(0)
                                    true_indexes = [(i + 1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                    for index in switch_relay:
                                        if index not in true_indexes:
                                            relay.state(index, on=True)
                                        relay.start_time[index] = time.time()
                                except Exception as e:
                                    print(f"⚠️ Relay operation failed: {e}. Continuing without relay control.")

                            if tracker_id not in violation_ids:
                                violation_ids.append(tracker_id)
                                xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                                datetimestamp = f"{datetime.now(tz).isoformat()}"
                                self.parent.create_result_events(
                                    xywh, obj_class, "Hazardous Area-Unsafe Zone",
                                    {"zone_name": zone_name}, datetimestamp, 1, self.parent.image
                                )

    # ─────────────────────────────────────────────────────────────────────────
    # PPE
    # ─────────────────────────────────────────────────────────────────────────
    def ppe(self):
        params       = self.parameters_data["PPE"]
        tz           = self.timezones["PPE"]
        relay        = self.relays["PPE"]
        switch_relay = self.switch_relays["PPE"]

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            ppe_objects    = params["subcategory_mapping"]  # dict: {"head": "No Helmet", ...}
            person_indices = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
            ppe_indices    = [i for i, cls in enumerate(self.parent.classes) if cls in ppe_objects.keys()]

            for idx in person_indices:
                box                    = self.parent.detection_boxes[idx]
                obj_class              = self.parent.classes[idx]
                tracker_id             = self.parent.tracker_ids[idx]
                anchor                 = self.parent.anchor_points_original[idx]
                person_detection_score = self.parent.detection_score[idx]

                if person_detection_score < 0.7:
                    continue

                for zone_name, zone_polygon in self.zone_data["PPE"].items():

                    zone_check_passed = (
                        not self.ppe_zone_check_enabled or
                        is_bottom_in_zone(anchor, zone_polygon)
                    )

                    if zone_check_passed:
                        person_poly = Polygon([
                            (box[0], box[1]), (box[0], box[3]),
                            (box[2], box[3]), (box[2], box[1])
                        ])

                        for ppe_idx in ppe_indices:
                            ppe_box        = self.parent.detection_boxes[ppe_idx]
                            ppe_obj_class  = self.parent.classes[ppe_idx]
                            ppe_confidence = self.parent.detection_score[ppe_idx]

                            if ppe_confidence < 0.7:
                                continue

                            if is_object_in_zone(ppe_box, person_poly):
                                if tracker_id not in self.running_data["PPE"][zone_name]:
                                    self.running_data["PPE"][zone_name][tracker_id] = {}

                                if ppe_obj_class not in self.running_data["PPE"][zone_name][tracker_id]:
                                    self.running_data["PPE"][zone_name][tracker_id][ppe_obj_class] = 1
                                else:
                                    self.running_data["PPE"][zone_name][tracker_id][ppe_obj_class] += 1

                                if self.running_data["PPE"][zone_name][tracker_id][ppe_obj_class] > params["frame_accuracy"]:

                                    if relay is not None and params["relay"] == 1:
                                        try:
                                            status = relay.state(0)
                                            true_indexes = [(i + 1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                            for index in switch_relay:
                                                if index not in true_indexes:
                                                    relay.state(index, on=True)
                                                relay.start_time[index] = time.time()
                                                print("Changed the Index")
                                        except Exception as e:
                                            print(f"⚠️ Relay operation failed: {e}. Continuing without relay control.")

                                    if ppe_obj_class not in self.violation_id_data["PPE"]:
                                        self.violation_id_data["PPE"][ppe_obj_class] = []

                                    if tracker_id not in self.violation_id_data["PPE"][ppe_obj_class]:
                                        self.violation_id_data["PPE"][ppe_obj_class].append(tracker_id)

                                        xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                                        now_local     = datetime.now(tz)
                                        fake_utc      = now_local.replace(tzinfo=pytz.utc)
                                        datetimestamp = f"{fake_utc.isoformat()}"

                                        subcategory    = ppe_objects[ppe_obj_class]  # "head" -> "No Helmet"
                                        event_metadata = {}
                                        if self.ppe_zone_check_enabled:
                                            event_metadata["zone_name"] = zone_name

                                        self.parent.create_result_events(
                                            xywh, obj_class, f"PPE-{subcategory}",
                                            event_metadata, datetimestamp,
                                            ppe_confidence, self.parent.image
                                        )

    # ─────────────────────────────────────────────────────────────────────────
    # TimeBasedUnauthorizedAccess
    # ─────────────────────────────────────────────────────────────────────────
    def time_based_unauthorized_access(self):
        params       = self.parameters_data["TimeBasedUnauthorizedAccess"]
        tz           = self.timezones["TimeBasedUnauthorizedAccess"]
        relay        = self.relays["TimeBasedUnauthorizedAccess"]
        switch_relay = self.switch_relays["TimeBasedUnauthorizedAccess"]

        print(self.parent.frame_monitor_count)

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            # mirrors raw: loop through scheduled_time
            for schedule in params["scheduled_time"]:
                time_start_end = schedule["time_start_end"]
                days_of_week   = schedule["days"]

                current_time     = datetime.now(tz).time()
                current_day_name = datetime.now(tz).strftime('%A')

                # mirrors raw: return if today not scheduled
                if current_day_name not in days_of_week:
                    return

                for time_range in time_start_end:
                    start_time_str, end_time_str = time_range
                    start_time = datetime.strptime(start_time_str, "%H:%M").time()
                    end_time   = datetime.strptime(end_time_str,   "%H:%M").time()

                    if start_time <= current_time <= end_time:
                        offender_indices = [
                            i for i, cls in enumerate(self.parent.classes)
                            if cls in params["subcategory_mapping"]
                        ]
                        violation_ids = self.violation_id_data["TimeBasedUnauthorizedAccess"]

                        for idx in offender_indices:
                            box        = self.parent.detection_boxes[idx]
                            obj_class  = self.parent.classes[idx]
                            tracker_id = self.parent.tracker_ids[idx]
                            anchor     = self.parent.anchor_points_original[idx]

                            for zone_name, zone_polygon in self.zone_data["TimeBasedUnauthorizedAccess"].items():
                                if is_bottom_in_zone(anchor, zone_polygon) and (
                                    tracker_id not in violation_ids or params["relay"] == 1
                                ):
                                    if tracker_id not in self.running_data["TimeBasedUnauthorizedAccess"][zone_name]:
                                        self.running_data["TimeBasedUnauthorizedAccess"][zone_name][tracker_id] = 1
                                    else:
                                        self.running_data["TimeBasedUnauthorizedAccess"][zone_name][tracker_id] += 1

                                    if self.running_data["TimeBasedUnauthorizedAccess"][zone_name][tracker_id] > params["frame_accuracy"]:

                                        if relay is not None and params["relay"] == 1:
                                            try:
                                                status = relay.state(0)
                                                true_indexes = [(i + 1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                                for index in switch_relay:
                                                    if index not in true_indexes:
                                                        relay.state(index, on=True)
                                                    relay.start_time[index] = time.time()
                                            except Exception as e:
                                                print(f"⚠️ Relay operation failed: {e}. Continuing without relay control.")

                                        if tracker_id not in violation_ids:
                                            violation_ids.append(tracker_id)
                                            xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                                            datetimestamp = f"{datetime.now(tz).isoformat()}"
                                            self.parent.create_result_events(
                                                xywh, obj_class, "Security-Unauthorized Access",
                                                {"zone_name": zone_name}, datetimestamp, 1, self.parent.image
                                            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # stary parking
    # ─────────────────────────────────────────────────────────────────────────
    def stray_parking(self):
        params       = self.parameters_data["StrayParking"]
        tz           = self.timezones["StrayParking"]
        relay        = self.relays["StrayParking"]
        switch_relay = self.switch_relays["StrayParking"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            vehicle_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in params["subcategory_mapping"]
            ]

            for idx in vehicle_indices:
                tracker_id = self.parent.tracker_ids[idx]
                anchor     = self.parent.anchor_points_original[idx]

                for zone_name, zone_polygon in self.zone_data["StrayParking"].items():
                    if is_bottom_in_zone(anchor, zone_polygon):
                        if tracker_id not in self.running_data["StrayParking"][zone_name]:
                            self.running_data["StrayParking"][zone_name][tracker_id] = 1
                        else:
                            self.running_data["StrayParking"][zone_name][tracker_id] += 1

                        if (
                            self.running_data["StrayParking"][zone_name][tracker_id] > params["frame_accuracy"]
                            and tracker_id not in self.violation_id_data["StrayParking"]
                        ):
                            self.violation_id_data["StrayParking"].append(tracker_id)
                            print("Car is parked in no parking area: ", tracker_id)

                            if params["relay"] == 1:
                                trigger_relay(relay, switch_relay)

                            box  = self.parent.detection_boxes[idx]
                            xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                            datetimestamp = datetime.now(tz).isoformat()

                            self.parent.create_result_events(
                                xywh, self.parent.classes[idx], "Stray_Parking",
                                {"zone_name": zone_name}, datetimestamp, 1, self.parent.image
                            )
                    else:
                        if tracker_id in self.running_data["StrayParking"][zone_name]:
                            del self.running_data["StrayParking"][zone_name][tracker_id]

    # ─────────────────────────────────────────────────────────────────────────
    # unauthorised_vehicle_area
    # ─────────────────────────────────────────────────────────────────────────                            
    def unauthorised_vehicle_area(self):
        params       = self.parameters_data["UnauthorisedVehicleArea"]
        tz           = self.timezones["UnauthorisedVehicleArea"]
        relay        = self.relays["UnauthorisedVehicleArea"]
        switch_relay = self.switch_relays["UnauthorisedVehicleArea"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            vehicle_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in params["subcategory_mapping"]
            ]

            for idx in vehicle_indices:
                tracker_id = self.parent.tracker_ids[idx]
                anchor     = self.parent.anchor_points_original[idx]

                for zone_name, zone_polygon in self.zone_data["UnauthorisedVehicleArea"].items():
                    if (
                        is_bottom_in_zone(anchor, zone_polygon)
                        and tracker_id not in self.violation_id_data["UnauthorisedVehicleArea"]
                    ):
                        self.violation_id_data["UnauthorisedVehicleArea"].append(tracker_id)

                        if params["relay"] == 1:
                            trigger_relay(relay, switch_relay)

                        box  = self.parent.detection_boxes[idx]
                        xywh = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                        datetimestamp = datetime.now(tz).isoformat()

                        self.parent.create_result_events(
                            xywh, self.parent.classes[idx], "Security-Unauthorized_Vehicle",
                            {"zone_name": zone_name}, datetimestamp, 1, self.parent.image
                        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # vehicle_congestion
    # ───────────────────────────────────────────────────────────────────────── 
    def vehicle_congestion(self):
        params       = self.parameters_data["VehicleCongestion"]
        tz           = self.timezones["VehicleCongestion"]
        relay        = self.relays["VehicleCongestion"]
        switch_relay = self.switch_relays["VehicleCongestion"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        current_time = time.time()

        vehicle_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls in params["subcategory_mapping"]
        ]

        for zone_name, zone_polygon in self.zone_data["VehicleCongestion"].items():
            zone_state = self.running_data["VehicleCongestion"][zone_name]

            # use shapely .bounds since zones are stored as Polygon objects
            minx, miny, maxx, maxy = zone_polygon.bounds
            zone_bbox = [minx, miny, maxx, maxy]

            vehicle_count = 0
            for idx in vehicle_indices:
                if calculate_iou(self.parent.detection_boxes[idx], zone_bbox) >= params["min_iou"]:
                    vehicle_count += 1

            if vehicle_count >= params["min_cluster_size"]:
                if zone_state["start_time"] is None:
                    zone_state["start_time"] = current_time

                duration = current_time - zone_state["start_time"]

                if duration >= params["duration"] and not zone_state["triggered"]:
                    zone_state["triggered"] = True

                    if params["relay"] == 1:
                        trigger_relay(relay, switch_relay)

                    datetimestamp = datetime.now(tz).isoformat()

                    self.parent.create_result_events(
                        None, "vehicle", "Vehicle_Congestion",
                        {"zone_name": zone_name, "vehicle_count": vehicle_count},
                        datetimestamp, 1, self.parent.image
                    )
            else:
                # congestion cleared — reset
                zone_state["start_time"] = None
                zone_state["triggered"]  = False

    # ─────────────────────────────────────────────────────────────────────────
    # min_max_worker_count
    # ─────────────────────────────────────────────────────────────────────────                 
    def min_max_worker_count(self):
        params       = self.parameters_data["MinMaxWorkerCount"]
        tz           = self.timezones["MinMaxWorkerCount"]
        relay        = self.relays["MinMaxWorkerCount"]
        switch_relay = self.switch_relays["MinMaxWorkerCount"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        min_workers     = params["min_workers"]
        max_workers     = params["max_workers"]
        min_iou         = params["min_iou"]
        required_frames = params["required_frames"]

        for zone_name, zone_polygon in self.zone_data["MinMaxWorkerCount"].items():
            zone_state = self.running_data["MinMaxWorkerCount"][zone_name]

            # use shapely .bounds since zones are stored as Polygon objects
            minx, miny, maxx, maxy = zone_polygon.bounds
            zone_bbox = [minx, miny, maxx, maxy]

            # count persons in zone via IoU
            worker_count = 0
            for i, cls in enumerate(self.parent.classes):
                if cls != "person":
                    continue
                if calculate_iou(self.parent.detection_boxes[i], zone_bbox) >= min_iou:
                    worker_count += 1

            # determine violation type
            violation_type = None
            if worker_count < min_workers:
                violation_type = "understaffed"
            elif worker_count > max_workers:
                violation_type = "overcrowded"

            # no violation — reset all state
            if violation_type is None:
                zone_state["active"]         = False
                zone_state["violation_type"] = None
                zone_state["start_time"]     = None
                zone_state["start_frame"]    = None
                zone_state["logged"]         = False
                continue

            # violation just started
            if not zone_state["active"]:
                zone_state["active"]         = True
                zone_state["violation_type"] = violation_type
                zone_state["start_time"]     = time.time()
                zone_state["start_frame"]    = self.parent.frame_monitor_count
                zone_state["logged"]         = False
                continue

            # violation continues — check frame threshold
            elapsed_frames = self.parent.frame_monitor_count - zone_state["start_frame"]

            if elapsed_frames >= required_frames and not zone_state["logged"]:
                zone_state["logged"] = True

                end_time  = time.time()
                end_frame = self.parent.frame_monitor_count

                if params["relay"] == 1:
                    trigger_relay(relay, switch_relay)

                datetimestamp = f"{datetime.now(tz).isoformat()}_{zone_name}"

                print(f"[MIN/MAX WORKER] {violation_type.upper()} in zone {zone_name} (count={worker_count})")

                self.parent.create_result_events(
                    None, "person", "MinMax_Worker_Violation",
                    {
                        "zone_name":      zone_name,
                        "violation_type": violation_type,
                        "worker_count":   worker_count,
                        "start_time":     zone_state["start_time"],
                        "end_time":       end_time,
                        "start_frame":    zone_state["start_frame"],
                        "end_frame":      end_frame
                    },
                    datetimestamp, 1, self.parent.image
                )

    # ─────────────────────────────────────────────────────────────────────────
    # wrong_lane
    # ─────────────────────────────────────────────────────────────────────────                 
    def wrong_lane(self):
        params       = self.parameters_data["WrongLane"]
        tz           = self.timezones["WrongLane"]
        relay        = self.relays["WrongLane"]
        switch_relay = self.switch_relays["WrongLane"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        tolerance       = params["tolerance"]
        required_frames = params["required_frames"]
        lane_config     = params["lane_config"]
        vehicle_classes = params["subcategory_mapping"]

        for i, cls in enumerate(self.parent.classes):
            if cls not in vehicle_classes:
                continue

            box        = self.parent.detection_boxes[i]
            tracker_id = self.parent.tracker_ids[i]
            confidence = self.parent.detection_score[i]

            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            center = (cx, cy)

            for zone_name, zone_polygon in self.zone_data["WrongLane"].items():
                if not is_object_in_zone(box, zone_polygon):
                    continue

                zone_state = self.running_data["WrongLane"][zone_name]

                if tracker_id not in zone_state:
                    zone_state[tracker_id] = {
                        "points":      deque(maxlen=params["history_length"]),
                        "start_frame": None
                    }

                state = zone_state[tracker_id]
                state["points"].append(center)

                # need at least 2 points to compute direction
                if len(state["points"]) < 2:
                    continue

                # compute total movement distance
                total_distance = 0
                for j in range(1, len(state["points"])):
                    dx = state["points"][j][0] - state["points"][j - 1][0]
                    dy = state["points"][j][1] - state["points"][j - 1][1]
                    total_distance += (dx**2 + dy**2) ** 0.5

                # ignore idle vehicles
                if total_distance < params["min_movement"]:
                    continue

                # compute direction angle
                p1 = state["points"][0]
                p2 = state["points"][-1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                angle = np.degrees(np.arctan2(dy, dx)) % 360

                expected_angle = lane_config[zone_name]["expected_angle"]
                diff           = abs(angle - expected_angle)
                angular_diff   = min(diff, 360 - diff)

                wrong = angular_diff > tolerance

                if wrong:
                    if state["start_frame"] is None:
                        state["start_frame"] = self.parent.frame_monitor_count
                    else:
                        elapsed = self.parent.frame_monitor_count - state["start_frame"]

                        if (
                            elapsed >= required_frames
                            and tracker_id not in self.violation_id_data["WrongLane"]
                        ):
                            self.violation_id_data["WrongLane"].append(tracker_id)

                            if params["relay"] == 1:
                                trigger_relay(relay, switch_relay)

                            xywh = xywh_original_percentage(
                                box, self.parent.original_width, self.parent.original_height
                            )
                            datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                            print(f"[WRONG LANE] Vehicle {tracker_id} in zone {zone_name}")

                            self.parent.create_result_events(
                                xywh, cls, "Wrong_Lane",
                                {"zone_name": zone_name},
                                datetimestamp, confidence, self.parent.image
                            )
                else:
                    # direction correct — reset frame counter
                    state["start_frame"] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # loitering
    # ─────────────────────────────────────────────────────────────────────────       
    def loitering(self):
        params       = self.parameters_data["Loitering"]
        tz           = self.timezones["Loitering"]
        relay        = self.relays["Loitering"]
        switch_relay = self.switch_relays["Loitering"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        person_indices  = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
        novest_indices  = [i for i, cls in enumerate(self.parent.classes) if cls == "no-vest"]
        violation_ids   = self.violation_id_data["Loitering"]
        processed       = []  # avoid duplicate processing per second per tracker

        for idx in person_indices:
            box        = self.parent.detection_boxes[idx]
            obj_class  = self.parent.classes[idx]
            tracker_id = self.parent.tracker_ids[idx]

            for zone_name, zone_polygon in self.zone_data["Loitering"].items():

                if tracker_id in processed:
                    continue

                if not is_object_in_zone(box, zone_polygon):
                    continue

                processed.append(tracker_id)

                person_poly = Polygon([
                    (box[0], box[1]), (box[0], box[3]),
                    (box[2], box[3]), (box[2], box[1])
                ])

                for novest_idx in novest_indices:
                    novest_box = self.parent.detection_boxes[novest_idx]

                    if not is_object_in_zone(novest_box, person_poly):
                        continue

                    if tracker_id in violation_ids:
                        continue

                    # extract color and validate uniform using helper utils
                    avg_color = extract_median_color(self.parent.image, novest_box)

                    if uniform_validation(avg_color, "staff",
                                          self.loitering_hsv_target,
                                          self.loitering_hsv_tolerance):
                        print(f"[LOITERING] Staff detected: {tracker_id}")
                        continue  # staff — skip, no violation

                    if uniform_validation(avg_color, "student",
                                          self.loitering_hsv_target,
                                          self.loitering_hsv_tolerance):
                        print(f"[LOITERING] Student detected: {tracker_id}")

                        loitering_data = self.running_data["Loitering"][zone_name]

                        if tracker_id not in loitering_data:
                            loitering_data[tracker_id] = 1
                        else:
                            loitering_data[tracker_id] += 1

                        if loitering_data[tracker_id] > params["loitering_time"]:
                            violation_ids.append(tracker_id)

                            if params["relay"] == 1:
                                trigger_relay(relay, switch_relay)

                            box_viol      = self.parent.detection_boxes[idx]
                            xywh          = xywh_original_percentage(box_viol, self.parent.original_width, self.parent.original_height)
                            datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                            print(f"[LOITERING] Violation: {obj_class} tracker {tracker_id} in {zone_name}")

                            self.parent.create_result_events(
                                xywh, obj_class, "Behavioural Analytics-Loitering",
                                {"zone_name": zone_name},
                                datetimestamp, 1, self.parent.image
                            )
                
    # ─────────────────────────────────────────────────────────────────────────
    # unauthorised_access
    # ───────────────────────────────────────────────────────────────────────── 
    def unauthorised_access(self):
        params       = self.parameters_data["UnauthorisedAccess"]
        tz           = self.timezones["UnauthorisedAccess"]
        relay        = self.relays["UnauthorisedAccess"]
        switch_relay = self.switch_relays["UnauthorisedAccess"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        # mirrors raw: allowed_class from subcategory_mapping in config
        allowed_class  = set(params.get("subcategory_mapping", []))
        person_indices = [i for i, cls in enumerate(self.parent.classes) if cls in allowed_class]
        violation_ids  = self.violation_id_data["UnauthorisedAccess"]
        processed      = []  # avoid duplicate processing per second per tracker

        for idx in person_indices:
            box        = self.parent.detection_boxes[idx]
            obj_class  = self.parent.classes[idx]
            tracker_id = self.parent.tracker_ids[idx]
            confidence = self.parent.detection_score[idx]

            for zone_name, zone_polygon in self.zone_data["UnauthorisedAccess"].items():

                if tracker_id in processed:
                    continue

                # mirrors raw: is_object_in_zone (full box check, not anchor point)
                if not is_object_in_zone(box, zone_polygon):
                    continue

                processed.append(tracker_id)

                zone_data = self.running_data["UnauthorisedAccess"][zone_name]

                if tracker_id not in zone_data:
                    zone_data[tracker_id] = 0
                zone_data[tracker_id] += 1

                if (
                    tracker_id not in violation_ids
                    and zone_data[tracker_id] > params["frame_accuracy"]
                ):
                    violation_ids.append(tracker_id)

                    if params["relay"] == 1:
                        trigger_relay(relay, switch_relay)

                    xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                    datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                    print(f"[UNAUTHORISED ACCESS] Violation: {obj_class} tracker {tracker_id} in {zone_name}")

                    self.parent.create_result_events(
                        xywh, obj_class, "Emergency Control-Unauthorised Access",
                        {"zone_name": zone_name},
                        datetimestamp, confidence, self.parent.image
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # idle_person
    # ───────────────────────────────────────────────────────────────────────── 
    def idle_person(self):
        params       = self.parameters_data["IdlePerson"]
        tz           = self.timezones["IdlePerson"]
        relay        = self.relays["IdlePerson"]
        switch_relay = self.switch_relays["IdlePerson"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        allowed_class  = set(params.get("subcategory_mapping", []))
        person_indices = [i for i, cls in enumerate(self.parent.classes) if cls in allowed_class]
        violation_ids  = self.violation_id_data["IdlePerson"]
        current_time   = time.time()

        for idx in person_indices:
            box        = self.parent.detection_boxes[idx]
            tracker_id = self.parent.tracker_ids[idx]
            confidence = self.parent.detection_score[idx]
            center_point = np.array(bottom_center(box))

            for zone_name, zone_polygon in self.zone_data["IdlePerson"].items():
                zone_data = self.running_data["IdlePerson"][zone_name]

                if tracker_id in zone_data and is_object_in_zone(box, zone_polygon):
                    # check movement from last position
                    movement = np.linalg.norm(center_point - zone_data[tracker_id]["last_pos"])

                    if movement < 5:
                        zone_data[tracker_id]["unmoving_time"] += current_time - zone_data[tracker_id]["last_time"]

                        if (
                            tracker_id not in violation_ids
                            and zone_data[tracker_id]["unmoving_time"] > params["idle_threshold"]
                        ):
                            violation_ids.append(tracker_id)

                            if params["relay"] == 1:
                                trigger_relay(relay, switch_relay)

                            xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                            datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                            print(f"[IDLE PERSON] idle person detected: {tracker_id}")

                            self.parent.create_result_events(
                                xywh, "person", "Behavioural Analytics-Idle Worker",
                                {"zone_name": zone_name},
                                datetimestamp, confidence, self.parent.image
                            )
                    else:
                        # person moved — reset timer
                        zone_data[tracker_id]["unmoving_time"] = 0

                    zone_data[tracker_id]["last_pos"]  = center_point
                    zone_data[tracker_id]["last_time"] = current_time

                else:
                    # first time seeing tracker or outside zone — init state
                    zone_data[tracker_id] = {
                        "last_pos":      center_point,
                        "last_time":     current_time,
                        "unmoving_time": 0
                    }
    

    # ─────────────────────────────────────────────────────────────────────────
    # running_detection
    # ───────────────────────────────────────────────────────────────────────── 
    def running_detection(self):
        params = self.parameters_data["RunningDetection"]
        tz     = self.timezones["RunningDetection"]

        if not activity_active_time(params, tz):
            return

        # runs every frame, no 1-second gate
        y_lines            = params.get("line_draw_y", [])
        known_distance_m   = params.get("known_distance_m", 2.0)
        speed_threshold    = params["speed_threshold"]
        max_history        = params["max_history"]
        current_frame      = self.parent.frame_monitor_count
        running_tracking   = self.running_data["RunningDetection"]
        violation_ids      = self.violation_id_data["RunningDetection"]

        # build calibration data from y_lines
        calibration_data = []
        for i in range(len(y_lines) - 1):
            y1 = y_lines[i]
            y2 = y_lines[i + 1]
            pixel_dist = abs(y2 - y1)
            if pixel_dist == 0:
                continue
            ppm   = pixel_dist / known_distance_m
            avg_y = (y1 + y2) / 2
            calibration_data.append({"y": avg_y, "ppm": ppm})
        calibration_data = sorted(calibration_data, key=lambda x: x["y"])

        for i, cls in enumerate(self.parent.classes):
            if cls != "person":
                continue

            box        = self.parent.detection_boxes[i]
            tracker_id = self.parent.tracker_ids[i]
            confidence = self.parent.detection_score[i]

            cx       = (box[0] + box[2]) / 2
            cy       = box[3]
            position = (cx, cy)

            # update track history
            history = running_tracking.setdefault(tracker_id, {"positions": [], "timestamps": []})
            history["positions"].append(position)
            history["timestamps"].append(time.time())

            if len(history["positions"]) > max_history:
                history["positions"]  = history["positions"][-max_history:]
                history["timestamps"] = history["timestamps"][-max_history:]

            if len(history["positions"]) < 3:
                continue

            # calculate instantaneous speed from last 3 positions
            recent_positions  = history["positions"][-3:]
            recent_timestamps = history["timestamps"][-3:]
            total_distance    = 0

            for j in range(1, len(recent_positions)):
                p1     = recent_positions[j - 1]
                p2     = recent_positions[j]
                px_dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                avg_y  = (p1[1] + p2[1]) / 2
                ppm    = get_interpolated_ppm(calibration_data, avg_y)
                total_distance += px_dist / ppm if ppm else 0

            time_diff     = recent_timestamps[-1] - recent_timestamps[0]
            instant_speed = total_distance / time_diff if time_diff > 0 else 0

            # violation check
            if instant_speed >= speed_threshold and tracker_id not in violation_ids:
                violation_ids[tracker_id] = current_frame

                xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                print(f"[RUNNING] ID={tracker_id}, Speed={instant_speed:.2f} m/s")

                self.parent.create_result_events(
                    xywh, cls, "Behavioural Analytics-Running",
                    {"speed": round(instant_speed, 2)},
                    datetimestamp, confidence, self.parent.image
                )

    # ─────────────────────────────────────────────────────────────────────────
    # vehicle_interaction
    # ───────────────────────────────────────────────────────────────────────── 
    def vehicle_interaction(self):
        params       = self.parameters_data["VehicleInteraction"]
        tz           = self.timezones["VehicleInteraction"]
        relay        = self.relays["VehicleInteraction"]
        switch_relay = self.switch_relays["VehicleInteraction"]

        if not activity_active_time(params, tz):
            return

        # runs every frame, no 1-second gate
        violation_ids   = self.violation_id_data["VehicleInteraction"]
        prev_pos        = self.running_data["VehicleInteraction"]["prev_pos"]
        prev_distances  = self.running_data["VehicleInteraction"]["prev_distances"]
        vehicle_classes = params.get("vehicle_classes", [])
        person_classes  = params.get("person_classes", [])

        vehicles       = []
        people         = []
        processed_ids  = []

        for idx in range(len(self.parent.detection_boxes)):
            box        = self.parent.detection_boxes[idx]
            obj_class  = self.parent.classes[idx]
            tracker_id = self.parent.tracker_ids[idx]

            if tracker_id in violation_ids:
                continue

            for zone_name, zone_polygon in self.zone_data["VehicleInteraction"].items():
                if not is_object_in_zone(box, zone_polygon) or tracker_id in processed_ids:
                    continue

                processed_ids.append(tracker_id)

                width  = box[2] - box[0]
                height = box[3] - box[1]
                anchor_point = (int((box[0] + box[2]) / 2), int(box[1] + height * 0.7))

                motion_vector = (0, 0)
                in_motion     = False

                if tracker_id in prev_pos:
                    prev_box   = prev_pos[tracker_id]
                    prev_h     = prev_box[3] - prev_box[1]
                    prev_anchor = (int((prev_box[0] + prev_box[2]) / 2), int(prev_box[1] + prev_h * 0.7))
                    dx = anchor_point[0] - prev_anchor[0]
                    dy = anchor_point[1] - prev_anchor[1]
                    displacement = math.hypot(dx, dy)
                    if displacement > params["motion_thr"]:
                        motion_vector = (dx, dy)
                        in_motion     = True

                obj_data = {
                    "tracker_id":   tracker_id,
                    "box":          box,
                    "class":        obj_class,
                    "zone":         zone_name,
                    "anchor_point": anchor_point,
                    "width":        width,
                    "height":       height,
                    "motion_vector": motion_vector,
                    "in_motion":    in_motion
                }

                if obj_class in vehicle_classes:
                    vehicles.append(obj_data)
                if obj_class in person_classes:
                    people.append(obj_data)

            prev_pos[tracker_id] = box

        # ── Vehicle-Person interactions ──
        for vehicle in vehicles:
            if not vehicle["in_motion"]:
                continue

            for person in people:
                if vehicle["zone"] != person["zone"]:
                    continue
                if vehicle["tracker_id"] in violation_ids and person["tracker_id"] in violation_ids:
                    continue

                v_point = vehicle["anchor_point"]
                p_point = person["anchor_point"]
                horizontal_dist = abs(v_point[0] - p_point[0])
                vertical_dist   = abs(v_point[1] - p_point[1])
                prox_h = (vehicle["width"] / 2 + person["width"] / 2) * (1 + params["hor_interaction_percentage"])
                prox_v = max(vehicle["height"], person["height"]) * params["ver_interaction_percentage"]
                proximity_violation = horizontal_dist < prox_h and vertical_dist < prox_v

                if not proximity_violation:
                    continue

                id_pair          = (min(vehicle["tracker_id"], person["tracker_id"]), max(vehicle["tracker_id"], person["tracker_id"]))
                current_distance = math.sqrt((v_point[0] - p_point[0]) ** 2 + (v_point[1] - p_point[1]) ** 2)

                if id_pair not in prev_distances:
                    prev_distances[id_pair] = current_distance
                    continue

                approaching = (prev_distances[id_pair] - current_distance) > 0
                prev_distances[id_pair] = current_distance

                if approaching and proximity_violation:
                    violation_ids.append(vehicle["tracker_id"])
                    violation_ids.append(person["tracker_id"])

                    if params["relay"] == 1:
                        trigger_relay(relay, switch_relay)

                    merged_box     = merge_box(vehicle["box"], person["box"])
                    xywh           = xywh_original_percentage(merged_box, self.parent.original_width, self.parent.original_height)
                    violation_type = "Approaching" if not person["in_motion"] else "Interaction"
                    datetimestamp  = f"{datetime.now(tz).isoformat()}_{vehicle['tracker_id']}_{person['tracker_id']}"

                    print(f"[VEHICLE INTERACTION] vehicle-person: {vehicle['tracker_id']} → {person['tracker_id']}")

                    self.parent.create_result_events(
                        xywh, person["class"],
                        f"Machine Control Area-Vehicle {violation_type}: {vehicle['class']} approaching {person['class']}",
                        {"zone_name": vehicle["zone"], "violation_type": violation_type, "person_moving": person["in_motion"]},
                        datetimestamp, 0.6, self.parent.image
                    )

        # ── Vehicle-Vehicle interactions ──
        if len(vehicles) > 1:
            for i in range(len(vehicles)):
                for j in range(len(vehicles)):
                    if i == j:
                        continue
                    v1, v2 = vehicles[i], vehicles[j]
                    if not (v1["in_motion"] or v2["in_motion"]):
                        continue
                    if v1["zone"] != v2["zone"]:
                        continue
                    if v1["tracker_id"] in violation_ids and v2["tracker_id"] in violation_ids:
                        continue

                    p1, p2          = v1["anchor_point"], v2["anchor_point"]
                    horizontal_dist = abs(p1[0] - p2[0])
                    vertical_dist   = abs(p1[1] - p2[1])
                    prox_h = (v1["width"] / 2 + v2["width"] / 2) * (1 + params.get("vv_hor_interaction_percentage", 0.2))
                    prox_v = max(v1["height"], v2["height"]) * params.get("vv_ver_interaction_percentage", 0.2)
                    proximity_violation = horizontal_dist < prox_h and vertical_dist < prox_v

                    if not proximity_violation:
                        continue

                    id_pair          = (min(v1["tracker_id"], v2["tracker_id"]), max(v1["tracker_id"], v2["tracker_id"]))
                    current_distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

                    if id_pair not in prev_distances:
                        prev_distances[id_pair] = current_distance
                        continue

                    approaching = (prev_distances[id_pair] - current_distance) > 0
                    prev_distances[id_pair] = current_distance

                    if approaching and proximity_violation:
                        violation_ids.append(v1["tracker_id"])
                        violation_ids.append(v2["tracker_id"])

                        if params["relay"] == 1:
                            trigger_relay(relay, switch_relay)

                        merged_box     = merge_box(v1["box"], v2["box"])
                        xywh           = xywh_original_percentage(merged_box, self.parent.original_width, self.parent.original_height)
                        violation_type = "Collision Risk" if v1["in_motion"] and v2["in_motion"] else "Approaching"
                        datetimestamp  = f"{datetime.now(tz).isoformat()}_{v1['tracker_id']}_{v2['tracker_id']}"

                        print(f"[VEHICLE INTERACTION] vehicle-vehicle: {v1['tracker_id']} ↔ {v2['tracker_id']}")

                        self.parent.create_result_events(
                            xywh, v2["class"],
                            f"Machine Control Area-Vehicle {violation_type}: {v1['class']} approaching {v2['class']}",
                            {"zone_name": v1["zone"], "violation_type": violation_type, "both_moving": v1["in_motion"] and v2["in_motion"]},
                            datetimestamp, 0.6, self.parent.image
                        )


    # ─────────────────────────────────────────────────────────────────────────
    # calibrate_light_histograms
    # ───────────────────────────────────────────────────────────────────────── 
    def _calibrate_light_histograms(self, params, tz):
        """
        Runs frame-by-frame calibration stages.
        Returns True only when calibration is fully complete.
        Mirrors raw calibrate_light_histograms() exactly.
        """
        cal_config = params["calibration_config"]
        wait_before_on  = cal_config["wait_before_on"]
        frames_on       = cal_config["frames_on"]
        wait_before_off = cal_config["wait_before_off"]
        frames_off      = cal_config["frames_off"]

        ld_data = self.running_data["LightDetection"]

        # use first zone polygon for calibration — mirrors raw
        zone_name, zone_polygon = list(self.zone_data["LightDetection"].items())[0]

        # use helper — passes self.parent.image (fixes original self.frame bug)
        frame_gray = get_zone_grayscale(self.parent.image, zone_polygon)
        if frame_gray is None:
            return False

        # init calibration state on first call
        if ld_data["calibration_state"] is None:
            ld_data["calibration_state"] = {
                "stage":        "wait_on",
                "frame_count":  0,
                "on_hist_list": [],
                "off_hist_list": []
            }
            print("[CALIBRATION] Starting. Please keep LIGHT ON.")

        state = ld_data["calibration_state"]
        state["frame_count"] += 1

        if state["stage"] == "wait_on":
            if state["frame_count"] >= wait_before_on:
                print("[CALIBRATION] Capturing ON histograms...")
                state["stage"]       = "calibrate_on"
                state["frame_count"] = 0

        elif state["stage"] == "calibrate_on":
            state["on_hist_list"].append(compute_histogram(frame_gray))
            if state["frame_count"] >= frames_on:
                print("[CALIBRATION] ON done. Please switch LIGHT OFF.")
                state["stage"]       = "wait_off"
                state["frame_count"] = 0

        elif state["stage"] == "wait_off":
            if state["frame_count"] >= wait_before_off:
                print("[CALIBRATION] Capturing OFF histograms...")
                state["stage"]       = "calibrate_off"
                state["frame_count"] = 0

        elif state["stage"] == "calibrate_off":
            state["off_hist_list"].append(compute_histogram(frame_gray))
            if state["frame_count"] >= frames_off:
                on_avg  = np.mean(state["on_hist_list"],  axis=0)
                off_avg = np.mean(state["off_hist_list"], axis=0)
                ld_data["calibration_data"]["ON"]  = on_avg
                ld_data["calibration_data"]["OFF"] = off_avg
                ld_data["calibration_done"]        = True
                print("[CALIBRATION] Completed successfully.")
                return True

        return False
        
    # ─────────────────────────────────────────────────────────────────────────
    # light_detection
    # ───────────────────────────────────────────────────────────────────────── 
    def light_detection(self):
        params = self.parameters_data["LightDetection"]
        tz     = self.timezones["LightDetection"]

        if not activity_active_time(params, tz):
            return

        ld_data = self.running_data["LightDetection"]

        # run calibration until complete — mirrors raw calibration_done check
        if not ld_data["calibration_done"]:
            self._calibrate_light_histograms(params, tz)
            return

        on_hist  = ld_data["calibration_data"]["ON"]
        off_hist = ld_data["calibration_data"]["OFF"]

        # parse working hours from config
        working_start = datetime.strptime(params["working_hour_start"], "%H:%M:%S").time()
        working_end   = datetime.strptime(params["working_hour_end"],   "%H:%M:%S").time()
        now           = datetime.now(tz).time()
        is_working    = working_start <= now < working_end

        violation_states = self.violation_id_data["LightDetection"]

        for zone_name, zone_polygon in self.zone_data["LightDetection"].items():

            # use helper — passes self.parent.image (fixes original self.frame bug)
            frame_gray = get_zone_grayscale(self.parent.image, zone_polygon)
            if frame_gray is None:
                return

            # compute histogram and compare to calibrated ON/OFF
            hist      = compute_histogram(frame_gray)
            on_score  = cv2.compareHist(hist, np.array(on_hist),  cv2.HISTCMP_CORREL)
            off_score = cv2.compareHist(hist, np.array(off_hist), cv2.HISTCMP_CORREL)

            current_state = "ON" if on_score > off_score else "OFF"

            # init zone violation state on first run
            vstate = violation_states.setdefault(zone_name, {})
            if "light_state" not in vstate:
                vstate["light_state"]            = current_state
                vstate["state_change_frame"]     = self.parent.frame_monitor_count
                vstate["violation_start_frame"]  = None
                print(f"[LIGHT DETECTION] Initial state in {zone_name}: {current_state}")

            # stable state change — only accept after 10 consistent frames
            if current_state != vstate["light_state"]:
                if self.parent.frame_monitor_count - vstate["state_change_frame"] >= 10:
                    vstate["light_state"]        = current_state
                    vstate["state_change_frame"] = self.parent.frame_monitor_count
                    print(f"[LIGHT DETECTION] State changed to {current_state} in {zone_name}")
            else:
                vstate["state_change_frame"] = self.parent.frame_monitor_count

            # determine if this is a violation
            violation = (
                (is_working     and vstate["light_state"] == "OFF") or
                (not is_working and vstate["light_state"] == "ON")
            )

            if violation:
                if vstate["violation_start_frame"] is None:
                    vstate["violation_start_frame"] = self.parent.frame_monitor_count
                    print(
                        f"[LIGHT DETECTION] Violation: light is '{vstate['light_state']}' "
                        f"in {zone_name} during {'working' if is_working else 'non-working'} hours"
                    )
            else:
                # violation just cleared — log it with duration
                if vstate["violation_start_frame"] is not None:
                    duration_frames  = self.parent.frame_monitor_count - vstate["violation_start_frame"]
                    duration_seconds = duration_frames / 20.0  # assumes ~20fps
                    timestamp        = now.isoformat()

                    print(
                        f"[LIGHT DETECTION] Violation ended in {zone_name}, "
                        f"lasted {round(duration_seconds, 2)}s, state: {vstate['light_state']}"
                    )

                    self.parent.create_result_events(
                        [0, 0, 0, 0], "light", "Light State Violation",
                        {
                            "zone_name":              zone_name,
                            "light_state":            vstate["light_state"],
                            "timestamp":              timestamp,
                            "violation_duration_sec": round(duration_seconds, 2),
                            "during_working_hour":    is_working
                        },
                        f"{timestamp}_light_{zone_name}",
                        1.0,
                        self.parent.image
                    )

                    vstate["violation_start_frame"] = None  # reset after logging
                        
    # ─────────────────────────────────────────────────────────────────────────
    # unattended_bag_detection
    # ───────────────────────────────────────────────────────────────────────── 
    def unattended_bag_detection(self):
        params       = self.parameters_data["unattended_bag_detection"]
        tz           = self.timezones["unattended_bag_detection"]
        relay        = self.relays["unattended_bag_detection"]
        switch_relay = self.switch_relays["unattended_bag_detection"]

        if not activity_active_time(params, tz):
            return

        bag_classes         = params["bag_classes"]
        UNATTENDED_THRESH_S = params["unattended_threshold_s"]
        BAG_MARGIN_FACTOR   = params["bag_margin_factor"]

        current_time  = time.time()
        processed     = []
        violation_ids = self.violation_id_data["unattended_bag_detection"]
        bag_data      = self.running_data["unattended_bag_detection"]

        person_indices = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
        bag_indices    = [i for i, cls in enumerate(self.parent.classes) if cls in bag_classes]

        for bag_idx in bag_indices:
            bag_box        = self.parent.detection_boxes[bag_idx]
            bag_tracker_id = self.parent.tracker_ids[bag_idx]

            for zone_name, zone_polygon in self.zone_data["unattended_bag_detection"].items():
                if bag_tracker_id in processed:
                    continue

                if not is_bottom_in_zone(self.parent.anchor_points_original[bag_idx], zone_polygon):
                    continue

                processed.append(bag_tracker_id)

                if bag_tracker_id in violation_ids:
                    continue

                # Calculate enlarged bag area for proximity check
                x1, y1, x2, y2 = bag_box
                w, h = x2 - x1, y2 - y1
                area = (
                    x1 - BAG_MARGIN_FACTOR * w,
                    y1 - BAG_MARGIN_FACTOR * h,
                    x2 + BAG_MARGIN_FACTOR * w,
                    y2 + BAG_MARGIN_FACTOR * h
                )

                # Init zone and tracker entry if first seen
                if zone_name not in bag_data:
                    bag_data[zone_name] = {}

                if bag_tracker_id not in bag_data[zone_name]:
                    bag_data[zone_name][bag_tracker_id] = {"last_attended": current_time}

                # Check if any person is near the bag
                someone_near = any(
                    area[0] <= (self.parent.detection_boxes[idx][0] + self.parent.detection_boxes[idx][2]) / 2 <= area[2]
                    and area[1] <= self.parent.detection_boxes[idx][3] <= area[3]
                    for idx in person_indices
                )

                if someone_near:
                    bag_data[zone_name][bag_tracker_id]["last_attended"] = current_time
                elif current_time - bag_data[zone_name][bag_tracker_id]["last_attended"] >= UNATTENDED_THRESH_S:
                    # Trigger violation once
                    violation_ids.append(bag_tracker_id)

                    if params["relay"] == 1:
                        trigger_relay(relay, switch_relay)

                    xywh = xywh_original_percentage(bag_box, self.parent.original_width, self.parent.original_height)
                    datetimestamp = f"{datetime.now(tz).isoformat()}_{bag_tracker_id}"

                    print(f"[UNATTENDED BAG] Violation: tracker_id={bag_tracker_id}, zone={zone_name}")

                    self.parent.create_result_events(
                        xywh,
                        self.parent.classes[bag_idx],
                        "Unattended-Bag",
                        {"zone_name": zone_name},
                        datetimestamp,
                        1,
                        self.parent.image
                    )
    # ─────────────────────────────────────────────────────────────────────────
    # blur_detection
    # ─────────────────────────────────────────────────────────────────────────
    def blur_detection(self):
        params = self.parameters_data["blur_detection"]
        tz     = self.timezones["blur_detection"]

        if not activity_active_time(params, tz):
            return

        threshold      = params["threshold"]
        gray           = cv2.cvtColor(self.parent.image, cv2.COLOR_BGR2GRAY)
        laplacian_var  = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"{laplacian_var:.2f}")
        if laplacian_var < threshold:
            current_time = time.time()
            blur_key     = f"blur_{self.parent.sensor_id}"
            last_alerts  = self.running_data["blur_detection"]

            if blur_key not in last_alerts or current_time - last_alerts[blur_key] > 5:
                last_alerts[blur_key] = current_time

                datetimestamp = f"{datetime.now(tz).isoformat()}_blur_detection"

                print(f"[BLUR DETECTED] Camera  - Laplacian variance: {laplacian_var:.2f}")

                self.parent.create_result_events(
                        [0, 0, 100, 100],
                        "camera",
                        "Blur-Detection",
                        {
                            "threshold": threshold,
                            "laplacian_variance": round(laplacian_var, 2),
                            "camera_id": self.parent.sensor_id
                        },
                        datetimestamp,
                        1,
                        self.parent.image
                    )
                    
    # ─────────────────────────────────────────────────────────────────────────
    # first_person_entering
    # ─────────────────────────────────────────────────────────────────────────
    def first_person_entering(self):
        params = self.parameters_data["first_person_entering"]
        tz     = self.timezones["first_person_entering"]
        if not activity_active_time(params, tz):
            return
        today       = datetime.now(tz).date()
        alert_dates = self.running_data["first_person_entering"]
        person_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls == "person"
        ]
        for zone_name, zone_polygon in self.zone_data["first_person_entering"].items():

            # Skip if alert already sent today for this zone
            if alert_dates.get(zone_name) == today:
                continue

            for idx in person_indices:
                anchor = self.parent.anchor_points_original[idx]

                if is_bottom_in_zone(anchor, zone_polygon):
                    tracker_id    = self.parent.tracker_ids[idx]
                    box           = self.parent.detection_boxes[idx]
                    xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                    datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                    print(f"[FIRST PERSON] ID: {tracker_id} entered '{zone_name}'. Sending alert.")

                    self.parent.create_result_events(
                        xywh,
                        "person",
                        "First-Person-Entering",
                        {"zone_name": zone_name},
                        datetimestamp,
                        1,
                        self.parent.image
                    )

                    # Mark alert sent for today, move to next zone
                    alert_dates[zone_name] = today
                    break
    # ─────────────────────────────────────────────────────────────────────────
    # last_person_entering
    # ─────────────────────────────────────────────────────────────────────────
    def last_person_leaving(self):
        params = self.parameters_data["last_person_leaving"]
        tz     = self.timezones["last_person_leaving"]

        now      = datetime.now(tz)
        now_time = now.time()
        today    = now.date()

        start_collection = datetime.strptime(params["start_collection_time"], "%H:%M:%S").time()
        trigger_alert    = datetime.strptime(params["trigger_alert_time"], "%H:%M:%S").time()

        zone_states = self.running_data["last_person_leaving"]

        for zone_name, zone_polygon in self.zone_data["last_person_leaving"].items():

            # Initialize zone state if first time seeing this zone
            if zone_name not in zone_states:
                zone_states[zone_name] = {
                    "last_info": None,
                    "alert_sent_date": None,
                    "collection_started": False
                }

            zone_state = zone_states[zone_name]

            # Reset state for new day
            # Reset state only if alert was sent on a PREVIOUS day
            if zone_state.get("alert_sent_date") is not None and zone_state["alert_sent_date"] != today:
                zone_state.update({
                    "alert_sent_date": None,
                    "last_info": None,
                    "collection_started": False
                })
            # Skip if alert already sent today
            elif zone_state.get("alert_sent_date") == today:
                continue
            
            # Add these prints temporarily
            print(f"[LAST PERSON DEBUG] now_time={now_time}, start_collection={start_collection}, trigger_alert={trigger_alert}")
            print(f"[LAST PERSON DEBUG] collection_started={zone_state.get('collection_started')}")
            print(f"[LAST PERSON DEBUG] alert_sent_date={zone_state.get('alert_sent_date')}, today={today}")
            # ── Phase 1: Collection Period ──
            if start_collection <= now_time < trigger_alert:
                zone_state["collection_started"] = True

                person_indices = [
                    i for i, cls in enumerate(self.parent.classes)
                    if cls == "person"
                ]

                for idx in person_indices:
                    anchor = self.parent.anchor_points_original[idx]

                    if is_bottom_in_zone(anchor, zone_polygon):
                        tracker_id = self.parent.tracker_ids[idx]
                        box        = self.parent.detection_boxes[idx]

                        zone_state["last_info"] = {
                            "box": box,
                            "tracker_id": tracker_id,
                            "image": self.parent.image.copy(),
                            "timestamp": now
                        }
                        print(f"[LAST PERSON] Zone '{zone_name}': tracker_id={tracker_id} recorded at {now.isoformat()}")

            # ── Phase 2: Trigger Period ──
            elif now_time >= trigger_alert and zone_state.get("collection_started"):
                print(f"[LAST PERSON] Trigger time reached for zone '{zone_name}'")

                if zone_state["last_info"] and not zone_state["alert_sent_date"]:
                    last_info  = zone_state["last_info"]
                    tracker_id = last_info["tracker_id"]
                    box        = last_info["box"]

                    xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                    datetimestamp = f"{now.isoformat()}_{tracker_id}"

                    print(f"[LAST PERSON] Last person ID={tracker_id} in '{zone_name}'. Sending alert.")

                    self.parent.create_result_events(
                        xywh,
                        "person",
                        "Last-Person-Leaving",
                        {
                            "zone_name": zone_name,
                            "last_seen_time": last_info["timestamp"].isoformat(),
                            "alert_trigger_time": now.isoformat()
                        },
                        datetimestamp,
                        1,
                        last_info["image"]
                    )

                    zone_state["alert_sent_date"] = today

                elif not zone_state["last_info"]:
                    print(f"[LAST PERSON] Trigger time reached but no one detected in '{zone_name}' during collection.")
                    zone_state["alert_sent_date"] = today
                    
    # ─────────────────────────────────────────────────────────────────────────
    # hairnet_detection
    # ─────────────────────────────────────────────────────────────────────────
    def hairnet_detection(self):
        params = self.parameters_data["hairnet_detection"]
        tz     = self.timezones["hairnet_detection"]

        if not activity_active_time(params, tz):
            return

        iou_threshold  = params["iou_threshold"]
        frame_accuracy = params["frame_accuracy"]

        person_indices  = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
        hairnet_indices = [i for i, cls in enumerate(self.parent.classes) if cls == "hairnet"]
        novest_indices  = [i for i, cls in enumerate(self.parent.classes) if cls == "no-vest"]

        processed          = []
        current_visible_ids = set()
        no_hairnet_counter = self.running_data["hairnet_detection"]
        violation_ids      = self.violation_id_data["hairnet_detection"]

        for idx in person_indices:
            box        = self.parent.detection_boxes[idx]
            tracker_id = self.parent.tracker_ids[idx]
            obj_class  = self.parent.classes[idx]
            anchor     = self.parent.anchor_points_original[idx]

            for zone_name, zone_polygon in self.zone_data["hairnet_detection"].items():
                if tracker_id in processed:
                    continue
                if not is_bottom_in_zone(anchor, zone_polygon):
                    continue

                # Convert person box to polygon for no-vest overlap check
                x1, y1, x2, y2 = box
                person_poly = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                # Check if person is staff via no-vest uniform color
                is_staff = False
                for novest_idx in novest_indices:
                    novest_box = self.parent.detection_boxes[novest_idx]
                    nv_anchor  = self.parent.anchor_points_original[novest_idx]
                    if is_bottom_in_zone(nv_anchor, person_poly):
                        avg_color = self.parent.extract_median_color(novest_box)
                        if self.parent.uniform_validation(avg_color, "staff"):
                            is_staff = True
                            break

                if not is_staff:
                    continue

                processed.append(tracker_id)
                current_visible_ids.add(tracker_id)

                # Define head box (top 20% of person box)
                head_box = [x1, y1, x2, y1 + 0.2 * (y2 - y1)]

                # Check hairnet overlap with head box
                hairnet_found = any(
                    calculate_iou(head_box, self.parent.detection_boxes[hidx]) > iou_threshold
                    for hidx in hairnet_indices
                )

                if hairnet_found:
                    no_hairnet_counter.pop(tracker_id, None)
                else:
                    current_count = no_hairnet_counter.get(tracker_id, 0) + 1
                    no_hairnet_counter[tracker_id] = current_count

                    if current_count >= frame_accuracy and tracker_id not in violation_ids:
                        violation_ids.append(tracker_id)

                        xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                        datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                        print(f"[HAIRNET] Violation: Staff ID={tracker_id} zone={zone_name}")

                        self.parent.create_result_events(
                            xywh,
                            obj_class,
                            "Hairnet-Violation",
                            {"zone_name": zone_name},
                            datetimestamp,
                            1,
                            self.parent.image
                        )
                        
    # ─────────────────────────────────────────────────────────────────────────
    # broomstick_detection
    # ─────────────────────────────────────────────────────────────────────────
    def broomstick_detection(self):
        params = self.parameters_data["broomstick_detection"]
        tz     = self.timezones["broomstick_detection"]

        if not activity_active_time(params, tz):
            return

        iob_threshold  = params["iob_threshold"]
        frame_accuracy = params["frame_accuracy"]

        broom_indices      = [i for i, cls in enumerate(self.parent.classes) if cls == "broomstick"]
        processed          = []
        broom_counter      = self.running_data["broomstick_detection"]
        violation_ids      = self.violation_id_data["broomstick_detection"]

        for bidx in broom_indices:
            box        = self.parent.detection_boxes[bidx]
            tracker_id = self.parent.tracker_ids[bidx]
            obj_class  = self.parent.classes[bidx]
            broom_poly = shapely_box(*box)

            for zone_name, zone_polygon in self.zone_data["broomstick_detection"].items():
                if tracker_id in processed:
                    continue
                if not broom_poly.intersects(zone_polygon):
                    continue

                processed.append(tracker_id)

                # IoB: intersection area / broom area
                intersection_area = broom_poly.intersection(zone_polygon).area
                broom_area        = broom_poly.area
                iob               = intersection_area / broom_area if broom_area > 0 else 0.0

                if iob > iob_threshold:
                    current_count              = broom_counter.get(tracker_id, 0) + 1
                    broom_counter[tracker_id]  = current_count

                    if current_count >= frame_accuracy and tracker_id not in violation_ids:
                        violation_ids.append(tracker_id)

                        xywh          = xywh_original_percentage(box, self.parent.original_width, self.parent.original_height)
                        datetimestamp = f"{datetime.now(tz).isoformat()}_{tracker_id}"

                        print(f"[BROOMSTICK] Violation: tracker_id={tracker_id} zone={zone_name}")

                        self.parent.create_result_events(
                            xywh,
                            obj_class,
                            "Broomstick-Violation",
                            {"zone_name": zone_name},
                            datetimestamp,
                            1,
                            self.parent.image
                        )
    # ─────────────────────────────────────────────────────────────────────────
    # unattended_student
    # ─────────────────────────────────────────────────────────────────────────
    def unattended_student_detection(self):
        params = self.parameters_data["unattended_student_detection"]
        tz     = self.timezones["unattended_student_detection"]

        if not activity_active_time(params, tz):
            return

        UNATTENDED_THRESH_S   = params["unattended_student_threshold_s"]
        STUDENT_MARGIN_FACTOR = params["student_margin_factor"]

        current_time  = time.time()
        processed     = []
        student_data  = self.running_data["unattended_student_detection"]
        violation_ids = self.violation_id_data["unattended_student_detection"]

        # Use trained "student" class directly — no uniform color check needed
        student_indices = [i for i, cls in enumerate(self.parent.classes) if cls == "student"]
        person_indices  = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]

        for student_idx in student_indices:
            student_box       = self.parent.detection_boxes[student_idx]
            student_tracker_id = self.parent.tracker_ids[student_idx]
            anchor            = self.parent.anchor_points_original[student_idx]

            for zone_name, zone_polygon in self.zone_data["unattended_student_detection"].items():
                if student_tracker_id in processed:
                    continue
                if not is_bottom_in_zone(anchor, zone_polygon):
                    continue

                processed.append(student_tracker_id)

                if student_tracker_id in violation_ids:
                    continue

                # Init zone and tracker entry if first seen
                if zone_name not in student_data:
                    student_data[zone_name] = {}

                if student_tracker_id not in student_data[zone_name]:
                    student_data[zone_name][student_tracker_id] = {"last_attended": current_time}

                # Calculate enlarged monitoring area around student
                x1, y1, x2, y2 = student_box
                w, h = x2 - x1, y2 - y1
                area = (
                    x1 - STUDENT_MARGIN_FACTOR * w,
                    y1 - STUDENT_MARGIN_FACTOR * h,
                    x2 + STUDENT_MARGIN_FACTOR * w,
                    y2 + STUDENT_MARGIN_FACTOR * h
                )

                # Check if any person (non-student) is near the student
                someone_near = any(
                    area[0] <= (self.parent.detection_boxes[idx][0] + self.parent.detection_boxes[idx][2]) / 2 <= area[2]
                    and area[1] <= self.parent.detection_boxes[idx][3] <= area[3]
                    for idx in person_indices
                )

                if someone_near:
                    student_data[zone_name][student_tracker_id]["last_attended"] = current_time
                else:
                    time_unattended = current_time - student_data[zone_name][student_tracker_id]["last_attended"]

                    if time_unattended >= UNATTENDED_THRESH_S:
                        violation_ids.append(student_tracker_id)

                        xywh          = xywh_original_percentage(student_box, self.parent.original_width, self.parent.original_height)
                        datetimestamp = f"{datetime.now(tz).isoformat()}_{student_tracker_id}"

                        print(f"[UNATTENDED STUDENT] ID={student_tracker_id} unattended for {time_unattended:.1f}s in zone={zone_name}")

                        self.parent.create_result_events(
                            xywh,
                            "student",
                            "Unattended-Student",
                            {
                                "zone_name": zone_name,
                                "unattended_duration": round(time_unattended, 1),
                                "threshold": UNATTENDED_THRESH_S
                            },
                            datetimestamp,
                            1,
                            self.parent.image
                        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # heatmap_overlay
    # ─────────────────────────────────────────────────────────────────────────
    def heatmap_overlay(self):
        params = self.parameters_data["Heatmap_overlay"]
        tz     = self.timezones["Heatmap_overlay"]

        state                  = self.running_data["Heatmap_overlay"]
        classes_for_heatmap    = params.get("classes_for_heatmap", ["person"])
        INTENSITY              = params.get("intensity", 1.0)
        RADIUS                 = params.get("radius", 40)
        is_currently_active    = activity_active_time(params, tz)

        if is_currently_active:

            # Initialize heatmap matrix on first active frame
            if state["heatmap_matrix"] is None:
                h, w, _ = self.parent.image.shape
                state["heatmap_matrix"] = np.zeros((h, w), dtype=np.float32)
                print(f"[HEATMAP] Initialized matrix for sensor {self.parent.sensor_id}")

            # Accumulate heat points from tracked objects
            for i, box in enumerate(self.parent.detection_boxes):
                if self.parent.classes[i] not in classes_for_heatmap:
                    continue

                center_x = int(box[0] + (box[2] - box[0]) / 2)
                bottom_y = int(box[3])

                for dy in range(-RADIUS, RADIUS + 1):
                    for dx in range(-RADIUS, RADIUS + 1):
                        distance = np.sqrt(dx * dx + dy * dy)
                        if distance <= RADIUS:
                            ny, nx = bottom_y + dy, center_x + dx
                            if (0 <= nx < state["heatmap_matrix"].shape[1] and
                                    0 <= ny < state["heatmap_matrix"].shape[0]):
                                state["heatmap_matrix"][ny, nx] += INTENSITY
            
            print(f"[HEATMAP] Matrix stats → "
              f"max={state['heatmap_matrix'].max():.2f} | "
              f"mean={state['heatmap_matrix'].mean():.4f} | "
              f"nonzero_pixels={np.count_nonzero(state['heatmap_matrix'])}")
              
            # Generate visual overlay
            blurred      = cv2.GaussianBlur(state["heatmap_matrix"], (31, 31), 0)
            clipped      = np.clip(blurred, 0, 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(clipped, cv2.COLORMAP_JET)
            overlay      = cv2.addWeighted(self.parent.image, 0.4, heatmap_color, 0.6, 0)

            state["heatmap_image_to_send"]      = overlay
            state["window_active_previously"]   = True

        elif not is_currently_active and state["window_active_previously"]:
            # Time window just ended — send the final heatmap image
            if state["heatmap_image_to_send"] is not None:
                print(f"[HEATMAP] Time window ended for {self.parent.sensor_id}. Sending final image.")

                datetimestamp = f"{datetime.now(tz).isoformat()}_heatmap"

                self.parent.create_result_events(
                    [0, 0, 100, 100],
                    "heatmap",
                    "Heatmap-Final",
                    {"time_window": params.get("scheduled_time")},
                    datetimestamp,
                    1,
                    state["heatmap_image_to_send"]
                )

                # Reset for next cycle
                state["heatmap_matrix"]          = None
                state["heatmap_image_to_send"]   = None

            state["window_active_previously"] = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Queue Analytics System
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # QUEUE ANALYTICS — helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _qa_is_person_in_zone(self, box, polygon):
        """Check if bottom-center of box is inside polygon"""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        bottom_y = y2
        from shapely.geometry import Point
        return polygon.contains(Point(center_x, bottom_y))

    def _qa_queue_length(self, zone_name):
        """Returns current confirmed queue length"""
        state         = self.running_data["queue_analytics"]
        queue_confirmed = state["queue_confirmed"]
        queue_data      = state["queue_data"]
        if zone_name not in queue_confirmed:
            return 0
        return sum(
            1 for tid in queue_confirmed[zone_name]
            if tid in queue_data[zone_name]
        )

    def _qa_calculate_service_time(self, tracker_id, zone_name):
        """Returns time spent in queue for specific person"""
        state      = self.running_data["queue_analytics"]
        queue_data = state["queue_data"]
        if zone_name not in queue_data or tracker_id not in queue_data[zone_name]:
            return None
        return time.time() - queue_data[zone_name][tracker_id]

    def _qa_average_wait_time(self, zone_name):
        """Returns estimated wait time for new person joining queue"""
        state           = self.running_data["queue_analytics"]
        service_history = state["service_history"]
        params          = self.parameters_data["queue_analytics"]
        default_service_time = params.get("default_service_time", 30)

        if zone_name not in service_history or len(service_history[zone_name]) == 0:
            return default_service_time

        avg_service  = sum(service_history[zone_name]) / len(service_history[zone_name])
        queue_len    = self._qa_queue_length(zone_name)
        being_served = len(state["counter_confirmed"][zone_name])
        return (queue_len + max(0, being_served - 1)) * avg_service

    def _qa_handle_queue_departures(self, zone_name, current_queue_ids,
                                      current_counter_ids, current_time,
                                      counter_transition_sec):
        """Handle people who left the queue"""
        state = self.running_data["queue_analytics"]
        try:
            left_queue = state["queue_confirmed"][zone_name] - current_queue_ids - current_counter_ids

            for tracker_id in left_queue:
                if tracker_id not in state["queue_exit_time"][zone_name]:
                    state["queue_exit_time"][zone_name][tracker_id] = current_time
                    print(f"[QUEUE] Person {tracker_id} left queue at {current_time:.1f}")

            expired_ids = []
            for tracker_id, exit_time in list(state["queue_exit_time"][zone_name].items()):
                if current_time - exit_time >= counter_transition_sec:
                    if tracker_id not in state["counter_visited"][zone_name]:
                        state["abandonment_count"][zone_name] += 1
                        if tracker_id in state["queue_data"][zone_name]:
                            wait_time = current_time - state["queue_data"][zone_name][tracker_id]
                            print(f"[QUEUE] ABANDONMENT: Person {tracker_id} after {wait_time:.1f}s")
                        else:
                            print(f"[QUEUE] ABANDONMENT: Person {tracker_id} (wait time unknown)")
                    expired_ids.append(tracker_id)

            for tracker_id in expired_ids:
                state["queue_exit_time"][zone_name].pop(tracker_id, None)
                state["queue_data"][zone_name].pop(tracker_id, None)
                state["queue_confirmed"][zone_name].discard(tracker_id)

        except Exception as e:
            print(f"[QUEUE] Error handling queue departures zone={zone_name}: {e}")

    def _qa_handle_counter_departures(self, zone_name, current_counter_ids):
        """Handle people who left the counter"""
        state = self.running_data["queue_analytics"]
        try:
            left_counter = set(state["counter_data"][zone_name].keys()) - current_counter_ids
            for tracker_id in left_counter:
                state["counter_data"][zone_name].pop(tracker_id, None)
                state["counter_confirmed"][zone_name].discard(tracker_id)
        except Exception as e:
            print(f"[QUEUE] Error handling counter departures zone={zone_name}: {e}")

    def _qa_update_queue_state(self, zone_name):
        """Update queue state for one zone"""
        state  = self.running_data["queue_analytics"]
        params = self.parameters_data["queue_analytics"]

        if zone_name not in self.zone_data["queue_analytics"]:
            return

        zone_config     = self.zone_data["queue_analytics"][zone_name]
        queue_polygon   = zone_config["queue_area"]
        counter_polygon = zone_config["counter_area"]
        current_time    = time.time()

        confidence_threshold  = params.get("confidence_threshold", 0.5)
        queue_confirm_sec     = params.get("queue_confirmation_sec", 3)
        counter_confirm_sec   = params.get("counter_confirmation_sec", 3)
        counter_transition_sec = params.get("counter_transition_sec", 10)

        person_indices     = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
        current_queue_ids   = set()
        current_counter_ids = set()

        for idx in person_indices:
            try:
                box        = self.parent.detection_boxes[idx]
                tracker_id = self.parent.tracker_ids[idx]
                confidence = self.parent.detection_score[idx]   # ← detection_score not confidence

                if confidence < confidence_threshold:
                    continue

                in_queue   = self._qa_is_person_in_zone(box, queue_polygon)
                in_counter = self._qa_is_person_in_zone(box, counter_polygon)

                # ── Queue processing ──
                if in_queue:
                    current_queue_ids.add(tracker_id)
                    if tracker_id not in state["queue_data"][zone_name]:
                        state["queue_data"][zone_name][tracker_id] = current_time

                    entry_time = state["queue_data"][zone_name][tracker_id]
                    if current_time - entry_time >= queue_confirm_sec:
                        state["queue_confirmed"][zone_name].add(tracker_id)

                # ── Counter processing ──
                if in_counter:
                    current_counter_ids.add(tracker_id)
                    if tracker_id not in state["counter_data"][zone_name]:
                        state["counter_data"][zone_name][tracker_id] = current_time

                    entry_time = state["counter_data"][zone_name][tracker_id]
                    if current_time - entry_time >= counter_confirm_sec:
                        if tracker_id not in state["counter_confirmed"][zone_name]:
                            state["counter_confirmed"][zone_name].add(tracker_id)
                            state["counter_visited"][zone_name].add(tracker_id)

                            service_time = self._qa_calculate_service_time(tracker_id, zone_name)
                            if service_time is not None:
                                state["service_history"][zone_name].append(service_time)
                                print(f"[QUEUE] SERVICE: Person {tracker_id} served after {service_time:.1f}s")

            except Exception as e:
                print(f"[QUEUE] Error processing person idx={idx} zone={zone_name}: {e}")
                continue

        self._qa_handle_queue_departures(
            zone_name, current_queue_ids, current_counter_ids,
            current_time, counter_transition_sec
        )
        self._qa_handle_counter_departures(zone_name, current_counter_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # QUEUE ANALYTICS — main method
    # ─────────────────────────────────────────────────────────────────────────

    def queue_analytics(self):
        params = self.parameters_data["queue_analytics"]
        tz     = self.timezones["queue_analytics"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        state = self.running_data["queue_analytics"]

        for zone_name in self.zone_data["queue_analytics"].keys():
            try:
                self._qa_update_queue_state(zone_name)

                queue_len          = self._qa_queue_length(zone_name)
                num_in_counter     = len(state["counter_confirmed"][zone_name])
                avg_wait           = self._qa_average_wait_time(zone_name)
                abandonment_count  = state["abandonment_count"].get(zone_name, 0)

                print(f"[QUEUE] zone={zone_name} | queue={queue_len} | "
                      f"counter={num_in_counter} | avg_wait={avg_wait:.1f}s | "
                      f"abandonments={abandonment_count}")

                self.parent.create_result_analytics(
                    f"{datetime.now(tz).isoformat()}",
                    "queue_analytics",
                    "person",
                    zone_name,
                    "queue_metrics",
                    {
                        "queue_length":      queue_len,
                        "average_wait_time": round(avg_wait, 1),
                        "num_in_counter":    num_in_counter,
                        "abandonment_count": abandonment_count,
                        "zone_name":         zone_name
                    }
                )

            except Exception as e:
                print(f"[QUEUE] Error processing zone={zone_name}: {e}")  
                

    # ─────────────────────────────────────────────────────────────────────────
    # entry_exit_wle_logs
    # ─────────────────────────────────────────────────────────────────────────
    def entry_exit_wle_logs(self):
        params  = self.parameters_data["EntryExitWLELogs"]
        tz      = self.timezones["EntryExitWLELogs"]
        data    = self.running_data["EntryExitWLELogs"]
        lc_data = data["lc_data"]

        if not activity_active_time(params, tz):
            return

        # mirrors raw: track entry/exit per class per zone every frame
        allowed_classes = params.get("subcategory_mapping", [])

        for idx, obj_class in enumerate(self.parent.classes):
            if obj_class not in allowed_classes:
                continue

            box        = self.parent.detection_boxes[idx]
            tracker_id = self.parent.tracker_ids[idx]

            for zone_name, (entry_polygon, exit_polygon) in self.entry_exit_zones.items():
                zone_data = data[zone_name]

                # entry check
                if is_object_in_zone(box, entry_polygon):
                    if tracker_id not in zone_data["entry_ids"][obj_class]:
                        zone_data["entry_ids"][obj_class].append(tracker_id)
                        zone_data["entry_count"][obj_class] += 1
                        lc_data["Entry"] += 1

                # exit check
                if is_object_in_zone(box, exit_polygon):
                    if tracker_id not in zone_data["exit_ids"][obj_class]:
                        zone_data["exit_ids"][obj_class].append(tracker_id)
                        zone_data["exit_count"][obj_class] += 1
                        lc_data["Exit"] += 1

        # mirrors raw: send analytics every frame_send_sec seconds then reset
        if time.time() - params["last_frame_time"] < params["frame_send_sec"]:
            return

        params["last_frame_time"] = time.time()
        print("[ENTRY EXIT WLE LOGS] Sending analytics...")

        datetimestamp = datetime.now(tz).isoformat()

        self.parent.create_result_analytics(
            datetimestamp, "ENTRY", "object_class", "zone", "ABSOLUTE", lc_data["Entry"]
        )
        self.parent.create_result_analytics(
            datetimestamp, "EXIT", "object_class", "zone", "ABSOLUTE", lc_data["Exit"]
        )

        # reset counts after sending — mirrors raw exactly
        lc_data["Entry"] = 0
        lc_data["Exit"]  = 0
        
    # ─────────────────────────────────────────────────────────────────────────
    # workforce_efficiency
    # ─────────────────────────────────────────────────────────────────────────
    def workforce_efficiency(self):
        params = self.parameters_data["WorkforceEfficiency"]
        tz     = self.timezones["WorkforceEfficiency"]
        data   = self.running_data["WorkforceEfficiency"]

        if not activity_active_time(params, tz):
            return

        # mirrors raw: runs every frame, no 1-second gate
        allowed_classes = set(params.get("subcategory_mapping", []))

        # count persons in each zone this frame
        for idx, obj_class in enumerate(self.parent.classes):
            if obj_class not in allowed_classes:
                continue

            box = self.parent.detection_boxes[idx]

            for zone_name, zone_polygon in self.zone_data["WorkforceEfficiency"].items():
                if is_object_in_zone(box, zone_polygon):
                    data[zone_name] += 1
                    break  # mirrors raw: count person once even if in multiple zones

        # increment frame counter
        data["current_frame_count"] += 1

        # send analytics when frame limit reached — mirrors raw exactly
        if data["current_frame_count"] >= data["send_data_frame_limit"]:
            for zone_name in self.zone_data["WorkforceEfficiency"].keys():
                result = data[zone_name] / data["send_data_frame_limit"]
                data[zone_name] = 0  # reset zone count

                print(f"[WORKFORCE EFFICIENCY] Zone: {zone_name}, Efficiency: {result:.3f}")

                self.parent.create_result_analytics(
                    data["timestamp"],
                    "WORKPLACE_EFFICIENCY",
                    "person",
                    zone_name,
                    "AVERAGE",
                    result
                )

            # reset counters after sending
            data["current_frame_count"] = 0
            data["timestamp"]           = datetime.now(tz).isoformat()
            data["time"]                = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # unattended_area
    # ─────────────────────────────────────────────────────────────────────────
    def unattended_area(self):
        params       = self.parameters_data["UnattendedArea"]
        tz           = self.timezones["UnattendedArea"]
        relay        = self.relays["UnattendedArea"]
        switch_relay = self.switch_relays["UnattendedArea"]

        if not activity_active_time(params, tz):
            return

        # runs every frame, no 1-second gate
        current_time         = time.time()
        unattended_threshold = params.get("unattended_threshold_s", 300)

        # get all security guard indices once
        security_guard_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls in ["security_guard", "security guard"]
        ]

        for zone_name, zone_polygon in self.zone_data["UnattendedArea"].items():
            zone_state = self.running_data["UnattendedArea"][zone_name]

            # check if any security guard is present in this zone
            security_guard_found = False
            for idx in security_guard_indices:
                box = self.parent.detection_boxes[idx]
                if is_object_in_zone(box, zone_polygon):
                    security_guard_found = True
                    zone_state["last_security_guard_seen"] = current_time
                    # reset alert if guard returns
                    if zone_state["alert_sent"]:
                        zone_state["alert_sent"] = False
                    break

            if security_guard_found:
                continue

            # no guard found — check how long zone has been unattended
            time_unattended = current_time - zone_state["last_security_guard_seen"]
            cooldown_until  = zone_state.get("cooldown_until", 0.0)

            if (
                time_unattended >= unattended_threshold
                and not zone_state["alert_sent"]
            ):
                # skip if still in cooldown
                if current_time < cooldown_until:
                    continue

                # use zone bounds for xywh — mirrors raw zone_polygon.bounds usage
                minx, miny, maxx, maxy = zone_polygon.bounds
                zone_box = [minx, miny, maxx, maxy]
                xywh     = xywh_original_percentage(zone_box, self.parent.original_width, self.parent.original_height)

                datetimestamp = f"{datetime.now(tz).isoformat()}_unattended_{zone_name}"

                print(f"[UNATTENDED AREA] No security guard in '{zone_name}' for {time_unattended:.1f}s")

                zone_state["alert_sent"] = True

                if params["relay"] == 1:
                    trigger_relay(relay, switch_relay)

                self.parent.create_result_events(
                    xywh, "zone", "Emergency Control-Unattended Area",
                    {
                        "zone_name":           zone_name,
                        "unattended_duration": round(time_unattended, 1),
                        "threshold":           unattended_threshold
                    },
                    datetimestamp, 0.9, self.parent.image
                )

                # set 1-hour cooldown after alert
                zone_state["cooldown_until"] = current_time + 3600.0


    # ─────────────────────────────────────────────────────────────────────────
    # YOLOE TEST LOGIC
    # ─────────────────────────────────────────────────────────────────────────
    def yoloe_test(self):
        params = self.parameters_data["YoloeTest"]
        tz     = self.timezones["YoloeTest"]

        if not activity_active_time(params, tz):
            return

        if time.time() - params["last_check_time"] <= 1:
            return
        params["last_check_time"] = time.time()

        yoloe_confidence_threshold = params.get("yoloe_confidence", 0.0)
        condition_labels           = params.get("condition_label", [])
        if isinstance(condition_labels, str):
            condition_labels = [condition_labels]

        print("\n[YOLOE TEST] ──────────────────────────────────────────")

        # ── 1. Hailo detections (independent) ────────────────────────────────── #
        offender_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls in params["subcategory_mapping"]
        ]

        if offender_indices:
            print(f"[YOLOE TEST] Hailo — {len(offender_indices)} person(s):")
            for idx in offender_indices:
                print(
                    f"   • tracker_id={self.parent.tracker_ids[idx]}  "
                    f"bbox={[round(v, 1) for v in self.parent.detection_boxes[idx]]}"
                )
        else:
            print("[YOLOE TEST] Hailo — no persons detected.")

        # ── 2. YOLOE detections (independent) ────────────────────────────────── #
        yoloe_result_data = None
        with self.parent.yoloe_lock:
            activity_data = self.parent.yoloe_results.get("YoloeTest")
            if activity_data and activity_data.get("result"):
                yoloe_result_data = activity_data["result"]

        if yoloe_result_data is None:
            print("[YOLOE TEST] YOLOE — waiting for first result...")
            print("[YOLOE TEST] ──────────────────────────────────────────")
            return

        text_dets   = [d for d in yoloe_result_data.get("detections", [])
                       if d.get("source", "text") == "text"]
        visual_dets = [d for d in yoloe_result_data.get("detections", [])
                       if d.get("source") == "visual"]

        # text pass
        valid_text = [d for d in text_dets
                      if d.get("prompt") in condition_labels
                      and d.get("confidence", 0) >= yoloe_confidence_threshold]
        if valid_text:
            print(f"[YOLOE TEST] YOLOE text — {len(valid_text)} detection(s):")
            for det in valid_text:
                print(
                    f"   • {det['prompt']}  conf={det['confidence']:.2f}  "
                    f"bbox={[round(v,1) for v in det['bounding_box']]}  "
                    f"polygon={'yes' if len(det.get('polygon',[])) >= 3 else 'no'}"
                )
        else:
            print("[YOLOE TEST] YOLOE text — no detections.")

        # visual pass
        valid_visual = [d for d in visual_dets
                        if d.get("confidence", 0) >= yoloe_confidence_threshold]
        if valid_visual:
            print(f"[YOLOE TEST] YOLOE visual — {len(valid_visual)} detection(s):")
            for det in valid_visual:
                print(
                    f"   • {det['prompt']}  conf={det['confidence']:.2f}  "
                    f"bbox={[round(v,1) for v in det['bounding_box']]}  "
                    f"polygon={'yes' if len(det.get('polygon',[])) >= 3 else 'no'}"
                )
        else:
            print("[YOLOE TEST] YOLOE visual — no detections.")

        print(
            f"[YOLOE TEST] Summary — hailo_persons={len(offender_indices)}  "
            f"text={len(valid_text)}  visual={len(valid_visual)}"
        )
        print("[YOLOE TEST] ──────────────────────────────────────────")
                                
    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    def run_all(self):
        """Called every frame by detection.py"""
        for method in self.active_methods:
            method()
    
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP 
    # ─────────────────────────────────────────────────────────────────────────
    def clean_all(self):
        """Called periodically by detection.py to prune stale trackers"""
        if "UnauthorisedArea" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            entry_times     = self.running_data["UnauthorisedArea"]["person_entry_times"]
            self.violation_id_data["UnauthorisedArea"] = [
                tid for tid in self.violation_id_data["UnauthorisedArea"]
                if tid in active_trackers
            ]
            keys_to_remove = [ k for k in entry_times.keys()
                if k.rsplit("_", 1)[-1] not in active_trackers
            ]
            for k in keys_to_remove:
                del entry_times[k]
        
        if "PeopleGathering" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            # Reset violation_ids so zones can re-trigger if gathering disperses and reforms
            self.violation_id_data["PeopleGathering"] = [
                z for z in self.violation_id_data["PeopleGathering"]
                if z in self.running_data["PeopleGathering"]
                and self.running_data["PeopleGathering"][z] > 0
            ]
            
        if "UnsafeZone" in self.parameters_data:
            self.violation_id_data["UnsafeZone"] = [
                tid for tid in self.violation_id_data["UnsafeZone"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["UnsafeZone"].keys():
                self.running_data["UnsafeZone"][zone_name] = {
                    k: v for k, v in self.running_data["UnsafeZone"][zone_name].items()
                    if k in active_trackers
                }

        if "PPE" in self.parameters_data:
            for acts in self.parameters_data["PPE"].get("subcategory_mapping", {}).keys():
                self.violation_id_data["PPE"][acts] = [
                    tid for tid in self.violation_id_data["PPE"][acts]
                    if tid in active_trackers
                ]
            for zone_name in self.zone_data["PPE"].keys():
                self.running_data["PPE"][zone_name] = {
                    k: v for k, v in self.running_data["PPE"][zone_name].items()
                    if k in active_trackers
                }

        if "TimeBasedUnauthorizedAccess" in self.parameters_data:
            self.violation_id_data["TimeBasedUnauthorizedAccess"] = [
                tid for tid in self.violation_id_data["TimeBasedUnauthorizedAccess"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["TimeBasedUnauthorizedAccess"].keys():
                self.running_data["TimeBasedUnauthorizedAccess"][zone_name] = {
                    k: v for k, v in self.running_data["TimeBasedUnauthorizedAccess"][zone_name].items()
                    if k in active_trackers
                }

        if "StrayParking" in self.parameters_data:
            self.violation_id_data["StrayParking"] = [
                tid for tid in self.violation_id_data["StrayParking"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["StrayParking"].keys():
                self.running_data["StrayParking"][zone_name] = {
                    k: v for k, v in self.running_data["StrayParking"][zone_name].items()
                    if k in active_trackers
                }
                
        if "UnauthorisedVehicleArea" in self.parameters_data:
            self.violation_id_data["UnauthorisedVehicleArea"] = [
                tid for tid in self.violation_id_data["UnauthorisedVehicleArea"]
                if tid in active_trackers
            ]
            
        if "VehicleCongestion" in self.parameters_data:
            pass
            
        if "MinMaxWorkerCount" in self.parameters_data:
            pass
            
        if "WrongLane" in self.parameters_data:
            self.violation_id_data["WrongLane"] = [
                tid for tid in self.violation_id_data["WrongLane"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["WrongLane"].keys():
                self.running_data["WrongLane"][zone_name] = {
                    tid: data
                    for tid, data in self.running_data["WrongLane"][zone_name].items()
                    if tid in active_trackers
                }
                
        if "Loitering" in self.parameters_data:
            self.violation_id_data["Loitering"] = [
                tid for tid in self.violation_id_data["Loitering"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["Loitering"].keys():
                self.running_data["Loitering"][zone_name] = {
                    k: v for k, v in self.running_data["Loitering"][zone_name].items()
                    if k in active_trackers
                }
                
        if "UnauthorisedAccess" in self.parameters_data:
            self.violation_id_data["UnauthorisedAccess"] = [
                tid for tid in self.violation_id_data["UnauthorisedAccess"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["UnauthorisedAccess"].keys():
                self.running_data["UnauthorisedAccess"][zone_name] = {
                    k: v for k, v in self.running_data["UnauthorisedAccess"][zone_name].items()
                    if k in active_trackers
                }
                
        if "IdlePerson" in self.parameters_data:
            self.violation_id_data["IdlePerson"] = [
                tid for tid in self.violation_id_data["IdlePerson"]
                if tid in active_trackers
            ]
            for zone_name in self.zone_data["IdlePerson"].keys():
                self.running_data["IdlePerson"][zone_name] = {
                    k: v for k, v in self.running_data["IdlePerson"][zone_name].items()
                    if k in active_trackers
                }
                
        if "RunningDetection" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            # violation_id_data is a dict here, not a list
            self.violation_id_data["RunningDetection"] = {
                k: v for k, v in self.violation_id_data["RunningDetection"].items()
                if k in active_trackers
            }
            self.running_data["RunningDetection"] = {
                k: v for k, v in self.running_data["RunningDetection"].items()
                if k in active_trackers
            }
            
        if "VehicleInteraction" in self.parameters_data:
            self.violation_id_data["VehicleInteraction"] = [
                tid for tid in self.violation_id_data["VehicleInteraction"]
                if tid in active_trackers
            ]
            self.running_data["VehicleInteraction"]["prev_pos"] = {
                k: v for k, v in self.running_data["VehicleInteraction"]["prev_pos"].items()
                if k in active_trackers
            }
            # prune prev_distances — remove pairs where either tracker is gone
            self.running_data["VehicleInteraction"]["prev_distances"] = {
                k: v for k, v in self.running_data["VehicleInteraction"]["prev_distances"].items()
                if k[0] in active_trackers and k[1] in active_trackers
            }
        
        if "LightDetection" in self.parameters_data:
            pass
        
        if "unattended_bag_detection" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            bag_data        = self.running_data["unattended_bag_detection"]

            # Prune stale tracker entries from each zone
            for zone_name in bag_data:
                stale = [tid for tid in bag_data[zone_name] if tid not in active_trackers]
                for tid in stale:
                    del bag_data[zone_name][tid]

            # Prune violation_ids so a re-appearing bag can be re-evaluated
            self.violation_id_data["unattended_bag_detection"] = [
                tid for tid in self.violation_id_data["unattended_bag_detection"]
                if tid in active_trackers
            ]
            
        if "hairnet_detection" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids

            # Prune stale counters
            stale = [tid for tid in self.running_data["hairnet_detection"] if tid not in active_trackers]
            for tid in stale:
                del self.running_data["hairnet_detection"][tid]

            # Prune violation ids
            self.violation_id_data["hairnet_detection"] = [
                tid for tid in self.violation_id_data["hairnet_detection"]
                if tid in active_trackers
            ]
        if "broomstick_detection" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids

            # Prune stale counters
            stale = [tid for tid in self.running_data["broomstick_detection"] if tid not in active_trackers]
            for tid in stale:
                del self.running_data["broomstick_detection"][tid]

            # Prune violation ids so re-appearing broom can re-trigger
            self.violation_id_data["broomstick_detection"] = [
                tid for tid in self.violation_id_data["broomstick_detection"]
                if tid in active_trackers
            ]
        
        if "unattended_student_detection" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids

            # Prune stale tracker entries from each zone
            student_data = self.running_data["unattended_student_detection"]
            for zone_name in student_data:
                stale = [tid for tid in student_data[zone_name] if tid not in active_trackers]
                for tid in stale:
                    del student_data[zone_name][tid]

            # Prune violation ids so re-appearing student can re-trigger
            self.violation_id_data["unattended_student_detection"] = [
                tid for tid in self.violation_id_data["unattended_student_detection"]
                if tid in active_trackers
            ]
            
        if "queue_analytics" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            state           = self.running_data["queue_analytics"]

            for zone_name in list(self.zone_data.get("queue_analytics", {}).keys()):
                # Prune queue_data
                stale = [tid for tid in state["queue_data"][zone_name] if tid not in active_trackers]
                for tid in stale:
                    state["queue_data"][zone_name].pop(tid, None)
                    state["queue_confirmed"][zone_name].discard(tid)
                    state["queue_exit_time"][zone_name].pop(tid, None)

                # Prune counter_data
                stale = [tid for tid in state["counter_data"][zone_name] if tid not in active_trackers]
                for tid in stale:
                    state["counter_data"][zone_name].pop(tid, None)
                    state["counter_confirmed"][zone_name].discard(tid)

                # counter_visited is intentional — keep forever (tracks who was ever served today)
                
        if "EntryExitWLELogs" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            for zone_name in self.entry_exit_zones.keys():
                zone_data = self.running_data["EntryExitWLELogs"][zone_name]
                for obj_class in self.parameters_data["EntryExitWLELogs"].get("subcategory_mapping", []):
                    zone_data["entry_ids"][obj_class] = [
                        tid for tid in zone_data["entry_ids"][obj_class]
                        if tid in active_trackers
                    ]
                    zone_data["exit_ids"][obj_class] = [
                        tid for tid in zone_data["exit_ids"][obj_class]
                        if tid in active_trackers
                    ]
                    
        if "WorkforceEfficiency" in self.parameters_data:
            pass
            
        if "UnattendedArea" in self.parameters_data:
            pass
            
        # YOLOE CLEAN (JUST TEST)
        # ── YoloeTest
        if "YoloeTest" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            entry_times     = self.running_data["YoloeTest"]["person_entry_times"]

            self.violation_id_data["YoloeTest"] = [
                tid for tid in self.violation_id_data["YoloeTest"]
                if tid in active_trackers
            ]
            # key format is now: "{tracker_id}_scaffold_{s_idx}"
            # so split on "_scaffold_" to get tracker_id
            keys_to_remove = [
                k for k in entry_times.keys()
                if k.split("_scaffold_")[0] not in active_trackers
            ]
            for k in keys_to_remove:
                del entry_times[k]
