import time
from datetime import datetime
import pytz
from shapely.geometry import Polygon

from activity_helper_utils import (
    is_bottom_in_zone,
    xywh_original_percentage,
    activity_active_time,
    init_relay,
    trigger_relay,
    relay_auto_off
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
            if "zones" in details:
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
                
            elif activity == "unattended_bag_detection":
                self.running_data["unattended_bag_detection"] = {}   # zone_name -> {tracker_id -> {last_attended}}
                self.violation_id_data["unattended_bag_detection"] = []
                self.active_methods.append(self.unattended_bag_detection)
                print(f"✅ unattended_bag_detection registered successfully.")
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
