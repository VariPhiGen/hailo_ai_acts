import time
from datetime import datetime
from shapely.geometry import Polygon, box as shapely_box

from activities.activity_helper_utils import (
    calculate_iou,
    xywh_original_percentage,
    init_relay,
    trigger_relay,
    activity_active_time
)

class VehicleCongestion:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters

        # Per-zone congestion state (USES running_data pattern)
        self.running_data = {}
        for zone_name in zone_data.keys():
            self.running_data[zone_name] = {
                "start_time": None,
                "triggered": False
            }

        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

    def run(self):
        # Schedule gate
        if not activity_active_time(self.parameters, self.parent.timezone):
            return

        # Rate limiting (same as other activities)
        if time.time() - self.parameters["last_check_time"] <= 1:
            return
        self.parameters["last_check_time"] = time.time()

        current_time = time.time()

        # Filter only vehicle classes
        vehicle_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls in self.parameters["subcategory_mapping"]
        ]

        for zone_name, zone_points in self.zone_data.items():
            zone_polygon = Polygon(zone_points)
            zone_state = self.running_data[zone_name]

            vehicle_count = 0

            # ---------------- COUNT VEHICLES WITH IOU ----------------
            # Precompute zone bounding box once
            zone_x_coords = [p[0] for p in zone_points]
            zone_y_coords = [p[1] for p in zone_points]

            zone_bbox = [
                min(zone_x_coords),  # x_min
                min(zone_y_coords),  # y_min
                max(zone_x_coords),  # x_max
                max(zone_y_coords)   # y_max
            ]

            for idx in vehicle_indices:
                vehicle_box = self.parent.detection_boxes[idx]  # [x1, y1, x2, y2]
                iou = calculate_iou(vehicle_box, zone_bbox)
                if iou >= self.parameters["min_iou"]:
                    vehicle_count += 1

            # ---------------- CONGESTION LOGIC ----------------
            if vehicle_count >= self.parameters["min_cluster_size"]:
                if zone_state["start_time"] is None:
                    zone_state["start_time"] = current_time

                duration = current_time - zone_state["start_time"]

                if (
                    duration >= self.parameters["duration"]
                    and not zone_state["triggered"]
                ):
                    zone_state["triggered"] = True

                    if self.parameters["relay"] == 1:
                        trigger_relay(self.relay, self.switch_relay)

                    datetimestamp = datetime.now(
                        self.parent.timezone
                    ).isoformat()

                    self.parent.create_result_events(
                        None,
                        "vehicle",
                        "Vehicle_Congestion",
                        {
                            "zone_name": zone_name,
                            "vehicle_count": vehicle_count
                        },
                        datetimestamp,
                        1,
                        self.parent.image
                    )
            else:
                # ---------------- RESET WHEN CONGESTION CLEARS ----------------
                zone_state["start_time"] = None
                zone_state["triggered"] = False

    def cleaning(self):
        # Nothing tracker-specific to clean (zone-level logic)
        pass
