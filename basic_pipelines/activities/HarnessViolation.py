import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone, activity_active_time
)


class HarnessViolation:
    def __init__(self, parent, zone_data, parameters):
        self.parent     = parent
        self.parameters = parameters
        self.zone_data  = zone_data

        # Tracker IDs for which a violation has already been sent to Kafka.
        # Once in here the person is NOT re-checked until they leave + re-enter.
        self.violation_sent_for: set = set()

        # Tracker IDs that YOLOE confirmed are compliant (harness + hook found).
        # They won't be re-checked until they leave + re-enter.
        self.compliant_ids: set = set()

        # Per-person rate-limit for YOLOE queries
        self.last_checked_time: dict = {}
        self.check_interval = 2.0   # seconds between YOLOE queries per person

        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        self.timezone     = pytz.timezone(self.timezone_str)

    # ------------------------------------------------------------------ #
    #  YOLOE RESULT CALLBACK
    #  Called by yoloe_handler after it finishes processing.
    #  Runs in a background thread — only use thread-safe operations.
    # ------------------------------------------------------------------ #
    def _on_yoloe_result(self, result: dict, metadata: dict):
        tracker_id = metadata.get("hailo_tracker_id")
        if tracker_id is None:
            return

        if result.get("violation"):
            # Mark so we stop hammering YOLOE for this person
            self.violation_sent_for.add(tracker_id)
            print(
                f"🔴 [HarnessViolation] Violation logged for Person {tracker_id}. "
                f"Will not re-check until they leave the frame.",
                flush=True,
            )
        else:
            # All items found — person is compliant
            self.compliant_ids.add(tracker_id)
            print(
                f"🟢 [HarnessViolation] Person {tracker_id} is COMPLIANT. "
                f"Stopping further checks.",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    #  MAIN RUN LOOP  (called every detection frame)
    # ------------------------------------------------------------------ #
    def run(self):
        if not activity_active_time(self.parameters, self.timezone):
            return

        if time.time() - self.parameters["last_check_time"] < 1:
            return
        self.parameters["last_check_time"] = time.time()

        current_time = time.time()

        if not (hasattr(self.parent, 'yoloe_handler') and self.parent.yoloe_handler):
            return  # YOLOE not available — nothing to do

        # ── Find persons detected by Hailo ───────────────────────────── #
        person_indices = [
            i for i, cls in enumerate(self.parent.classes)
            if cls in self.parameters.get("subcategory_mapping", ["person"])
        ]

        for idx in person_indices:
            tracker_id = self.parent.tracker_ids[idx]
            anchor     = self.parent.anchor_points_original[idx]
            box        = self.parent.detection_boxes[idx]  # [xmin, ymin, xmax, ymax]

            for zone_name, zone_polygon in self.zone_data.items():
                if not is_bottom_in_zone(anchor, zone_polygon):
                    continue

                # Already processed this person — skip
                if tracker_id in self.violation_sent_for:
                    continue
                if tracker_id in self.compliant_ids:
                    continue

                # Per-person YOLOE query rate-limit
                last = self.last_checked_time.get(tracker_id, 0)
                if (current_time - last) < self.check_interval:
                    continue

                self.last_checked_time[tracker_id] = current_time

                meta = {
                    "sensor_id":         self.parent.sensor_id,
                    "activity":          "HarnessViolation",
                    "hailo_tracker_id":  tracker_id,
                    "rule":              "must_detect",
                    "required_items":    ["harness", "hook"],
                    "target_bbox":       box,
                }

                print(
                    f"🔍 [HarnessViolation] Querying YOLOE for Person {tracker_id} "
                    f"in {zone_name} ...",
                    flush=True,
                )

                self.parent.yoloe_handler.trigger(
                    frame=self.parent.image,
                    activity_name="HarnessViolation",
                    metadata=meta,
                    on_result=self._on_yoloe_result,  # ← closes the feedback loop
                )

    # ------------------------------------------------------------------ #
    #  CLEANING  (called periodically to free memory for gone persons)
    # ------------------------------------------------------------------ #
    def cleaning(self):
        active = self.parent.last_n_frame_tracker_ids

        self.last_checked_time = {
            k: v for k, v in self.last_checked_time.items()
            if k in active
        }
        self.violation_sent_for = {
            tid for tid in self.violation_sent_for
            if tid in active
        }
        self.compliant_ids = {
            tid for tid in self.compliant_ids
            if tid in active
        }
