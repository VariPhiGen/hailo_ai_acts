import time
from datetime import datetime
from activities.activity_helper_utils import (
    is_object_in_zone,
    xywh_original_percentage,
    init_relay,
    trigger_relay,
    activity_active_time
)
from shapely.geometry import Polygon

class Tailgating:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters

        # Global dedupe (same meaning as other activities)
        self.violation_id_data = []

        # Per-zone temporal state (REUSES running_data pattern)
        self.running_data = {}
        for zone_name in zone_data.keys():
            self.running_data[zone_name] = {
                "active": False,
                "frame_count": 0,
                "initiator": None,
                "violators": set()
            }

        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

    def run(self):
        # Schedule gate (shared utility)
        if not activity_active_time(self.parameters, self.parent.timezone):
            return

        # Rate limiting (same as old activities)
        if time.time() - self.parameters["last_check_time"] <= 1:
            return
        self.parameters["last_check_time"] = time.time()

        # Only persons
        person_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls == "person"
        ]

        for zone_name, zone_polygon_pts in self.zone_data.items():
            zone_polygon = Polygon(zone_polygon_pts)
            zone_state = self.running_data[zone_name]

            # Collect persons currently inside the zone
            persons_in_zone = []

            for idx in person_indices:
                box = self.parent.detection_boxes[idx]
                tracker_id = self.parent.tracker_ids[idx]

                if is_object_in_zone(box, zone_polygon):
                    persons_in_zone.append((idx, tracker_id))

            # ---------------- NO ONE IN ZONE → RESET ----------------
            if len(persons_in_zone) == 0:
                zone_state["active"] = False
                zone_state["frame_count"] = 0
                zone_state["initiator"] = None
                zone_state["violators"].clear()
                continue

            # ---------------- START COUNTDOWN ----------------
            if not zone_state["active"]:
                idx, tracker_id = persons_in_zone[0]
                zone_state["active"] = True
                zone_state["frame_count"] = 0
                zone_state["initiator"] = tracker_id
                zone_state["violators"].clear()
                continue

            # ---------------- COUNTDOWN ACTIVE ----------------
            zone_state["frame_count"] += 1

            # Timeout → reset
            if zone_state["frame_count"] > self.parameters["max_frames"]:
                zone_state["active"] = False
                zone_state["frame_count"] = 0
                zone_state["initiator"] = None
                zone_state["violators"].clear()
                continue

            # ---------------- DETECT TAILGATERS ----------------
            for idx, tracker_id in persons_in_zone:
                if tracker_id == zone_state["initiator"]:
                    continue

                if tracker_id in zone_state["violators"]:
                    continue

                if tracker_id in self.violation_id_data:
                    continue

                # TAILGATING VIOLATION
                zone_state["violators"].add(tracker_id)
                self.violation_id_data.append(tracker_id)

                if self.parameters["relay"] == 1:
                    trigger_relay(self.relay, self.switch_relay)

                box = self.parent.detection_boxes[idx]
                xywh = xywh_original_percentage(box)
                datetimestamp = datetime.now(
                    self.parent.timezone
                ).isoformat()

                print(
                    f"[TAILGATING] Zone={zone_name}, Initiator={zone_state['initiator']}, "
                    f"Violator={tracker_id}"
                )

                self.parent.create_result_events(
                    xywh,
                    "person",
                    "Tailgating",
                    {"zone_name": zone_name},
                    datetimestamp,
                    1,
                    self.parent.image
                )

    def cleaning(self):
        self.violation_id_data = [
            tid for tid in self.violation_id_data
            if tid in self.parent.last_n_frame_tracker_ids
        ]
