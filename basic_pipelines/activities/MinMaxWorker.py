import time
from datetime import datetime

from activities.activity_helper_utils import (
    calculate_iou,
    xywh_original_percentage,
    init_relay,
    trigger_relay,
    activity_active_time
)

class MinMaxWorkerCount:
    def __init__(self, parent, zone_data, parameters):
        self.parent = parent
        self.zone_data = zone_data
        self.parameters = parameters

        # Per-zone persistent violation state (NEW concept, must be initialized)
        self.running_data = {}
        for zone_name in zone_data.keys():
            self.running_data[zone_name] = {
                "active": False,
                "violation_type": None,   # "understaffed" | "overcrowded"
                "start_time": None,
                "start_frame": None,
                "logged": False
            }

        self.relay, self.switch_relay = init_relay(self.parent, self.parameters)

    def run(self):
        # Schedule gate
        if not activity_active_time(self.parameters, self.parent.timezone):
            return

        # Rate limit (same pattern as old activities)
        if time.time() - self.parameters["last_check_time"] <= 1:
            return
        self.parameters["last_check_time"] = time.time()

        min_workers = self.parameters["min_workers"]
        max_workers = self.parameters["max_workers"]
        min_iou = self.parameters["min_iou"]
        required_frames = self.parameters["required_frames"]

        for zone_name, zone_points in self.zone_data.items():
            zone_state = self.running_data[zone_name]

            # Convert polygon → bbox once per run (IoU utility expects bbox)
            xs = [p[0] for p in zone_points]
            ys = [p[1] for p in zone_points]
            zone_bbox = [min(xs), min(ys), max(xs), max(ys)]

            # ---------------- COUNT PERSONS ----------------
            worker_count = 0

            for i, cls in enumerate(self.parent.classes):
                if cls != "person":
                    continue

                box = self.parent.detection_boxes[i]
                iou = calculate_iou(box, zone_bbox)

                if iou >= min_iou:
                    worker_count += 1

            # ---------------- DETERMINE VIOLATION TYPE ----------------
            violation_type = None
            if worker_count < min_workers:
                violation_type = "understaffed"
            elif worker_count > max_workers:
                violation_type = "overcrowded"

            # ---------------- NO VIOLATION → RESET ----------------
            if violation_type is None:
                zone_state["active"] = False
                zone_state["violation_type"] = None
                zone_state["start_time"] = None
                zone_state["start_frame"] = None
                zone_state["logged"] = False
                continue

            # ---------------- VIOLATION START ----------------
            if not zone_state["active"]:
                zone_state["active"] = True
                zone_state["violation_type"] = violation_type
                zone_state["start_time"] = time.time()
                zone_state["start_frame"] = self.parent.frame_monitor_count
                zone_state["logged"] = False
                continue

            # ---------------- VIOLATION CONTINUES ----------------
            elapsed_frames = (
                self.parent.frame_monitor_count - zone_state["start_frame"]
            )

            if elapsed_frames >= required_frames and not zone_state["logged"]:
                zone_state["logged"] = True

                end_time = time.time()
                end_frame = self.parent.frame_monitor_count

                if self.parameters["relay"] == 1:
                    trigger_relay(self.relay, self.switch_relay)

                datetimestamp = (
                    f"{datetime.now(self.parent.timezone).isoformat()}_{zone_name}"
                )

                print(
                    f"[MIN/MAX WORKER] {violation_type.upper()} in zone {zone_name} "
                    f"(count={worker_count})"
                )

                self.parent.create_result_events(
                    None,
                    "person",
                    "MinMax_Worker_Violation",
                    {
                        "zone_name": zone_name,
                        "violation_type": violation_type,
                        "worker_count": worker_count,
                        "start_time": zone_state["start_time"],
                        "end_time": end_time,
                        "start_frame": zone_state["start_frame"],
                        "end_frame": end_frame
                    },
                    datetimestamp,
                    1,
                    self.parent.image
                )

    def cleaning(self):
        # Zone-level logic → nothing tracker-specific to clean
        pass
