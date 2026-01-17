import time
from datetime import datetime
from activities.activity_helper_utils import (
    is_bottom_in_zone,
    is_activity_active,
    xywh_original_percentage
)

class StrayParking:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters
        self.running_data = {}
        self.violation_id_data = []

        self.last_check_time = parameters.get("last_check_time", 0)

        self.relay = None
        if parameters.get("relay", 0) == 1:
            try:
                self.relay = self.parent.relay_handler
                self.switch_relay = parameters["switch_relay"]
            except Exception:
                self.relay = None

        for zone in zone_data:
            self.running_data[zone] = {}

    def run(self):
        if not is_activity_active(self.parameters, self.parent.timezone):
            return

        if time.time() - self.parameters["last_check_time"] > 1:
            self.parameters["last_check_time"] = time.time()

            for i, cls in enumerate(self.parent.classes):
                if cls not in self.parameters["subcategory_mapping"]:
                    continue

                tracker_id = self.parent.tracker_ids[i]
                anchor = self.parent.anchor_points_original[i]
                box = self.parent.detection_boxes[i]

                for zone_name, polygon in self.zone_data.items():
                    if is_bottom_in_zone(anchor, polygon):
                        if tracker_id not in self.running_data[zone_name]:
                            self.running_data[zone_name][tracker_id] = time.time()

                        elapsed = time.time() - self.running_data[zone_name][tracker_id]

                        if (
                            elapsed >= self.parameters["time_limit"]
                            and tracker_id not in self.violation_id_data
                        ):
                            self.violation_id_data.append(tracker_id)

                            if self.relay:
                                try:
                                    for r in self.switch_relay:
                                        self.relay.state(r, on=True)
                                        self.relay.start_time[r] = time.time()
                                except Exception:
                                    pass

                            xywh = xywh_original_percentage(
                                box,
                                self.parent.original_width,
                                self.parent.original_height
                            )
                            datetimestamp = datetime.now(self.parent.timezone).isoformat()

                            self.parent.create_result_events(
                                xywh,
                                cls,
                                "Stray Parking",
                                {"zone_name": zone_name},
                                datetimestamp,
                                1,
                                self.parent.image
                            )

    def cleaning(self):
        active = self.parent.last_n_frame_tracker_ids
        for zone in self.running_data:
            self.running_data[zone] = {
                k: v for k, v in self.running_data[zone].items()
                if k in active
            }
        self.violation_id_data = [
            v for v in self.violation_id_data if v in active
        ]
