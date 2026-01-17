import time
from datetime import datetime
from activities.activity_helper_utils import (
    is_bottom_in_zone,
    is_activity_active,
    xywh_original_percentage
)

class PeopleGathering:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters
        self.running_data = {}
        self.violation_id_data = []
        self.relay=None

        self.last_check_time = parameters.get("last_check_time", 0)

        #Initialize Relay
        if parameters["relay"]==1:
            try:
                if self.parent.relay_handler.device==None:
                    success = self.parent.relay_handler.initiate_relay()
                    if not success:
                        print("⚠️ Relay device not available. Continuing without relay control.")
                        self.relay = None
                self.relay=self.parent.relay_handler
                self.switch_relay=parameters["switch_relay"]
            except Exception:
                self.relay = None

        for zone in zone_data:
            self.running_data[zone] = 0  # frame counter

    def run(self):
        if not is_activity_active(self.parameters, self.parent.timezone):
            return

        if time.time() - self.parameters["last_check_time"] > 1:
            self.parameters["last_check_time"] = time.time()

            person_indices = [
                i for i, c in enumerate(self.parent.classes)
                if c == "person"
            ]

            for zone_name, polygon in self.zone_data.items():
                count = 0
                for idx in person_indices:
                    anchor = self.parent.anchor_points_original[idx]
                    if is_bottom_in_zone(anchor, polygon):
                        count += 1

                if count >= self.parameters["person_limit"]:
                    self.running_data[zone_name] += 1
                else:
                    self.running_data[zone_name] = 0

                if (
                    self.running_data[zone_name] > self.parameters["frame_accuracy"]
                    and zone_name not in self.violation_id_data
                ):
                    self.violation_id_data.append(zone_name)

                ####### make a utility fucntion for relay ##############################
                    if self.relay:
                        try:
                            for r in self.switch_relay:
                                self.relay.state(r, on=True)
                                self.relay.start_time[r] = time.time()
                        except Exception:
                            pass

                    datetimestamp = datetime.now(self.parent.timezone).isoformat()
                    self.parent.create_result_events(
                        None,
                        "person",
                        "People Gathering",
                        {"zone_name": zone_name, "count": count},
                        datetimestamp,
                        1,
                        self.parent.image
                    )

    def cleaning(self):
        pass
