import time
import numpy as np
from collections import deque
from datetime import datetime

from activities.activity_helper_utils import (
    is_object_in_zone,
    xywh_original_percentage,
    init_relay,
    trigger_relay,
    activity_active_time
)

class WrongLane:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters

        self.running_data = {}
        for zone_name in zone_data.keys():
            self.running_data[zone_name] = {}

        self.violation_id_data = []

        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

    def run(self):
        # Schedule gate
        if not activity_active_time(self.parameters, self.parent.timezone):
            return

        # Rate limit
        if time.time() - self.parameters["last_check_time"] <= 1:
            return
        self.parameters["last_check_time"] = time.time()

        tolerance = self.parameters["tolerance"]
        required_frames = self.parameters["required_frames"]
        lane_config = self.parameters["lane_config"]
        vehicle_classes = self.parameters["subcategory_mapping"]

        for i, cls in enumerate(self.parent.classes):
            if cls not in vehicle_classes:
                continue

            box = self.parent.detection_boxes[i]
            tracker_id = self.parent.tracker_ids[i]
            confidence = self.parent.detection_score[i]

            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            center = (cx, cy)

            for zone_name, polygon in self.zone_data.items():
                if not is_object_in_zone(box, polygon):
                    continue

                zone_state = self.running_data[zone_name]

                if tracker_id not in zone_state:
                    zone_state[tracker_id] = {
                        "points": deque(maxlen=self.parameters["history_length"]),
                        "start_frame": None
                    }

                state = zone_state[tracker_id]
                state["points"].append(center)

                # Need at least 2 points
                if len(state["points"]) < 2:
                    continue

                # Compute total movement distance
                total_distance = 0
                for j in range(1, len(state["points"])):
                    dx = state["points"][j][0] - state["points"][j - 1][0]
                    dy = state["points"][j][1] - state["points"][j - 1][1]
                    total_distance += (dx**2 + dy**2) ** 0.5

                # Ignore idle vehicles
                if total_distance < self.parameters["min_movement"]:
                    continue

                # Direction angle
                p1 = state["points"][0]
                p2 = state["points"][-1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                angle = np.degrees(np.arctan2(dy, dx)) % 360

                expected_angle = lane_config[zone_name]["expected_angle"]
                diff = abs(angle - expected_angle)
                angular_diff = min(diff, 360 - diff)

                wrong = angular_diff > tolerance

                if wrong:
                    if state["start_frame"] is None:
                        state["start_frame"] = self.parent.frame_monitor_count
                    else:
                        elapsed = (
                            self.parent.frame_monitor_count - state["start_frame"]
                        )

                        if (
                            elapsed >= required_frames
                            and tracker_id not in self.violation_id_data
                        ):
                            self.violation_id_data.append(tracker_id)

                            if self.parameters["relay"] == 1:
                                trigger_relay(self.relay, self.switch_relay)

                            xywh = xywh_original_percentage(
                                box,
                                self.parent.original_width,
                                self.parent.original_height
                            )

                            datetimestamp = (
                                f"{datetime.now(self.parent.timezone).isoformat()}_{tracker_id}"
                            )

                            print(
                                f"[WRONG LANE] Vehicle {tracker_id} in zone {zone_name}"
                            )

                            self.parent.create_result_events(
                                xywh,
                                cls,
                                "Wrong_Lane",
                                {"zone_name": zone_name},
                                datetimestamp,
                                confidence,
                                self.parent.image
                            )
                else:
                    # Reset if direction becomes correct
                    state["start_frame"] = None

    def cleaning(self):
        # Standard cleanup
        self.violation_id_data = [
            tid for tid in self.violation_id_data
            if tid in self.parent.last_n_frame_tracker_ids
        ]

        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                tid: data
                for tid, data in self.running_data[zone_name].items()
                if tid in self.parent.last_n_frame_tracker_ids
            }
