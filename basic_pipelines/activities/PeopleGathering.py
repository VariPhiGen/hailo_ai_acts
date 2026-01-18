import time
from datetime import datetime
from activities.activity_helper_utils import (
    is_bottom_in_zone,
    xywh_original_percentage,
    init_relay,
    trigger_relay,
    activity_active_time
)

class PeopleGathering:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters

        self.running_data = {}      # zone_name -> frame_count
        self.violation_id_data = []

        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

    def run(self):
        if not activity_active_time(self.parameters, self.parent.timezone):
            return

        if time.time() - self.parameters["last_check_time"] > 1:
            self.parameters["last_check_time"] = time.time()

            person_indices = [
                i for i, cls in enumerate(self.parent.classes)
                if cls == "person"
            ]

            for zone_name, zone_polygon in self.zone_data.items():
                count = 0

                for idx in person_indices:
                    anchor = self.parent.anchor_points_original[idx]
                    if is_bottom_in_zone(anchor, zone_polygon):
                        count += 1

                if count >= self.parameters["max_people"]:
                    self.running_data[zone_name] = self.running_data.get(zone_name, 0) + 1
                else:
                    self.running_data[zone_name] = 0

                if (
                    self.running_data[zone_name] >= self.parameters["frame_accuracy"]
                    and zone_name not in self.violation_id_data
                ):
                    self.violation_id_data.append(zone_name)

                    if self.parameters["relay"] == 1:
                        trigger_relay(self.relay, self.switch_relay)

                    datetimestamp = datetime.now(self.parent.timezone).isoformat()
                    self.parent.create_result_events(
                        None,
                        "person",
                        "People_Gathering",
                        {"zone_name": zone_name, "count": count},
                        datetimestamp,
                        1,
                        self.parent.image
                    )

    def cleaning(self):
        pass
