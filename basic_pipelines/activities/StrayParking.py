import time
from datetime import datetime
from activities.activity_helper_utils import (
    is_bottom_in_zone,
    xywh_original_percentage,
    init_relay,
    trigger_relay,
    activity_active_time
)

class StrayParking:
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
        if not activity_active_time(self.parameters, self.parent.timezone):
            return

        if time.time() - self.parameters["last_check_time"] > 1:
            self.parameters["last_check_time"] = time.time()

            vehicle_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls in self.parameters["subcategory_mapping"]
            ]

            for idx in vehicle_indices:
                tracker_id = self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]

                for zone_name, zone_polygon in self.zone_data.items():
                    if is_bottom_in_zone(anchor, zone_polygon):
                        if tracker_id not in self.running_data[zone_name]:
                            self.running_data[zone_name][tracker_id] = 1
                        else:
                            self.running_data[zone_name][tracker_id] += 1

                        if (
                            self.running_data[zone_name][tracker_id]
                            > self.parameters["frame_accuracy"]
                            and tracker_id not in self.violation_id_data
                        ):
                            self.violation_id_data.append(tracker_id)

                            if self.parameters["relay"] == 1:
                                trigger_relay(self.relay, self.switch_relay)

                            box = self.parent.detection_boxes[idx]
                            xywh = xywh_original_percentage(box)
                            datetimestamp = datetime.now(
                                self.parent.timezone
                            ).isoformat()

                            self.parent.create_result_events(
                                xywh,
                                self.parent.classes[idx],
                                "Stray_Parking",
                                {"zone_name": zone_name},
                                datetimestamp,
                                1,
                                self.parent.image
                            )
                    else:
                        if tracker_id in self.running_data[zone_name]:
                            del self.running_data[zone_name][tracker_id]

    def cleaning(self):
        self.violation_id_data = [
            tid for tid in self.violation_id_data
            if tid in self.parent.last_n_frame_tracker_ids
        ]

        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                k: v for k, v in self.running_data[zone_name].items()
                if k in self.parent.last_n_frame_tracker_ids
            }
