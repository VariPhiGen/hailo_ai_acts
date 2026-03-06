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
                # Mirrors raw file:
                #   self.person_entry_times = {}
                #   self.violation_id_data  = []
                self.running_data["UnauthorisedArea"] = {
                    "person_entry_times": {}
                }
                self.violation_id_data["UnauthorisedArea"] = []
                self.active_methods.append(self.unauthorised_area)
                print(f"✅ UnauthorisedArea registered successfully.")

    # ─────────────────────────────────────────────────────────────────────────
    # UnauthorisedArea — logic mirrors raw UnauthorisedArea.py run() exactly
    # ─────────────────────────────────────────────────────────────────────────
    def unauthorised_area(self):
        params       = self.parameters_data["UnauthorisedArea"]
        tz           = self.timezones["UnauthorisedArea"]
        relay        = self.relays["UnauthorisedArea"]
        switch_relay = self.switch_relays["UnauthorisedArea"]

        # mirrors raw: if not activity_active_time(self.parameters, self.timezone): return
        if not activity_active_time(params, tz):
            return

        # mirrors raw: if time.time()-self.parameters["last_check_time"]>1:
        if time.time() - params["last_check_time"] > 1:
            params["last_check_time"] = time.time()

            current_time  = time.time()
            entry_times   = self.running_data["UnauthorisedArea"]["person_entry_times"]
            violation_ids = self.violation_id_data["UnauthorisedArea"]

            # mirrors raw: offender_indices filtered by subcategory_mapping
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
                        # mirrors raw print
                        # print("person found in zone")

                        if zone_tracker_key not in entry_times:
                            # mirrors raw: record entry time and skip
                            entry_times[zone_tracker_key] = current_time
                            continue

                        time_in_zone = current_time - entry_times[zone_tracker_key]

                        # mirrors raw violation check exactly
                        if (
                            time_in_zone > params["time_limit"]
                            and tracker_id not in violation_ids
                        ):
                            violation_ids.append(tracker_id)
                            print("Violation found: ", tracker_id)

                            # mirrors raw: trigger relay if relay==1
                            if params["relay"] == 1:
                                trigger_relay(relay, switch_relay)

                            # mirrors raw: xywh_original_percentage(box) — no width/height args
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
                        # mirrors raw: person left zone → delete entry time
                        if zone_tracker_key in entry_times:
                            del entry_times[zone_tracker_key]

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    def run_all(self):
        """Called every frame by detection.py"""
        for method in self.active_methods:
            method()

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP — mirrors raw cleaning() exactly
    # ─────────────────────────────────────────────────────────────────────────
    def clean_all(self):
        """Called periodically by detection.py to prune stale trackers"""
        if "UnauthorisedArea" in self.parameters_data:
            active_trackers = self.parent.last_n_frame_tracker_ids
            entry_times     = self.running_data["UnauthorisedArea"]["person_entry_times"]
            # mirrors raw: prune violation_id_data list
            self.violation_id_data["UnauthorisedArea"] = [
                tid for tid in self.violation_id_data["UnauthorisedArea"]
                if tid in active_trackers
            ]
            # mirrors raw cleaning(): rsplit on zone_tracker_key to get tracker_id
            keys_to_remove = [
                k for k in entry_times.keys()
                if k.rsplit("_", 1)[-1] not in active_trackers
            ]
            for k in keys_to_remove:
                del entry_times[k]
