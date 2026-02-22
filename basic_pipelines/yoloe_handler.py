import threading
import requests
import cv2
import time
import json
import logging
from datetime import datetime

YOLOE_SERVER_URL = "http://localhost:5000/predict"

class YOLOEHandler:
    def __init__(self, config):
        self.config = config.get("yoloe_control", {})
        self.enabled = self.config.get("enabled", 0)
        self.target_activities = set(self.config.get("activities", []))
        self.cooldown = self.config.get("cooldown_sec", 2)

        # ── Key fix: cooldown is now per (activity, tracker_id) ──────── #
        self.last_trigger_time = {}   # key: (activity_name, tracker_id)

        self.results_queue = None

        # ── Circuit-breaker for YOLOE server failures ─────────────────── #
        self._failure_count = 0
        self._pause_until = 0.0
        self._lock = threading.Lock()

    def set_results_queue(self, queue):
        self.results_queue = queue

    def should_trigger(self, activity_name, tracker_id="global"):
        if not self.enabled:
            return False
        if activity_name not in self.target_activities:
            return False
        # Circuit-breaker: are we in a pause window?
        if time.time() < self._pause_until:
            return False
        now = time.time()
        key = (activity_name, tracker_id)
        if (now - self.last_trigger_time.get(key, 0)) < self.cooldown:
            return False
        return True

    def trigger(self, frame, activity_name, metadata=None, on_result=None):
        """
        on_result: optional callable(result_dict, metadata) invoked after
                   _handle_result(), letting the calling activity update its
                   own state (e.g. mark violation_sent_for).
        """
        tracker_id = metadata.get("hailo_tracker_id", "global") if metadata else "global"
        if not self.should_trigger(activity_name, tracker_id):
            return

        key = (activity_name, tracker_id)
        self.last_trigger_time[key] = time.time()

        # ── Crop to person bbox before sending for better accuracy ────── #
        send_frame = frame
        if metadata and "target_bbox" in metadata:
            try:
                xmin, ymin, xmax, ymax = [int(v) for v in metadata["target_bbox"]]
                h, w = frame.shape[:2]
                # Clamp to frame bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)
                crop = frame[ymin:ymax, xmin:xmax]
                if crop.size > 0:
                    send_frame = crop
            except Exception:
                pass  # fall back to full frame

        t = threading.Thread(
            target=self._send_request,
            args=(send_frame.copy(), frame.copy(), activity_name, metadata, on_result)
        )
        t.daemon = True
        t.start()

    def _send_request(self, send_frame, full_frame, activity_name, metadata, on_result):
        try:
            _, img_encoded = cv2.imencode('.jpg', send_frame)
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(YOLOE_SERVER_URL, files=files, timeout=10.0)

            if response.status_code == 200:
                # Success — reset failure counter
                with self._lock:
                    self._failure_count = 0
                result = response.json()
                self._handle_result(result, activity_name, metadata, full_frame, on_result)
            else:
                logging.error(f"YOLOE Server Error: {response.status_code}")
                self._record_failure()

        except requests.exceptions.Timeout:
            logging.warning(f"YOLOE Request Timed Out for {metadata.get('hailo_tracker_id','?')}")
            self._record_failure()
        except Exception as e:
            logging.error(f"YOLOE Request Failed: {e}")
            self._record_failure()

    def _record_failure(self):
        """Exponential back-off circuit breaker."""
        with self._lock:
            self._failure_count += 1
            pause = min(10 * (2 ** (self._failure_count - 1)), 300)  # cap at 5 min
            self._pause_until = time.time() + pause
            logging.warning(f"YOLOE: {self._failure_count} failures. Pausing for {pause}s")

    def _handle_result(self, result, activity_name, metadata, frame, on_result=None):
        detections = result.get("detections", [])
        detected_classes = [d['class'] for d in detections]

        rule           = metadata.get('rule', 'report_all')
        required_items = metadata.get('required_items', [])
        tracker_id     = metadata.get('hailo_tracker_id', 'unknown')

        messages_to_send = []

        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            h, w = frame.shape[:2]
        except Exception as e:
            logging.error(f"Failed to encode image: {e}")
            return

        # ── RULE: must_detect ─────────────────────────────────────────── #
        if rule == 'must_detect':
            found_items  = {item: False for item in required_items}
            found_confs  = {}

            for det in detections:
                cls = det['class']
                if cls in found_items:
                    found_items[cls] = True
                    found_confs[cls] = det['confidence']

            # Terminal summary — always shown so operator can see live state
            status_parts = []
            for item in required_items:
                if found_items[item]:
                    status_parts.append(
                        f"✅ {item.upper()} (conf: {found_confs[item]:.2f})"
                    )
                else:
                    status_parts.append(f"❌ {item.upper()} MISSING")

            print(
                f"[HarnessCheck] Person {tracker_id} → "
                + " | ".join(status_parts),
                flush=True
            )

            missing_items = [item for item in required_items if not found_items[item]]

            # Build violation event only when something is missing
            if missing_items:
                print(
                    f"⚠️  VIOLATION: Person {tracker_id} missing: "
                    f"{', '.join(missing_items)}",
                    flush=True
                )
                target_bbox = metadata.get('target_bbox', [0, 0, w, h])
                xmin, ymin, xmax, ymax = target_bbox
                xywh_pct = [
                    (xmin * 100) / w,
                    (ymin * 100) / h,
                    ((xmax - xmin) * 100) / w,
                    ((ymax - ymin) * 100) / h,
                ]
                formatted_bbox = [{
                    "xywh":        xywh_pct,
                    "class_name":  "violation",
                    "subcategory": f"Missing {','.join(missing_items)}",
                    "confidence":  1.0,
                    "parameters":  {},
                    "anpr":        "False",
                }]
                messages_to_send.append({
                    "type":       "VIOLATION_EVENT",
                    "tracker_id": str(tracker_id),
                    "bbox_data":  formatted_bbox,
                    "missing":    missing_items,   # pass back to callback
                })
            else:
                print(
                    f"✅ Person {tracker_id} — All required items present. No violation.",
                    flush=True
                )

        # ── RULE: must_not_detect ─────────────────────────────────────── #
        elif rule == 'must_not_detect':
            for i, det in enumerate(detections):
                if det['class'] in required_items:
                    messages_to_send.append({
                        "type":       "VIOLATION_EVENT",
                        "tracker_id": f"yoloe_{i}",
                        "bbox_data":  self._format_bbox(det, w, h),
                    })

        # ── RULE: report_all (default) ────────────────────────────────── #
        else:
            for i, det in enumerate(detections):
                messages_to_send.append({
                    "type":       "YOLOE_DETECTION",
                    "tracker_id": f"yoloe_{i}",
                    "bbox_data":  self._format_bbox(det, w, h),
                })

        # ── Push to Kafka queue ───────────────────────────────────────── #
        for msg in messages_to_send:
            if self.results_queue:
                full_message = {
                    "sensor_id":               metadata.get("sensor_id", "Unknown"),
                    "type":                    msg["type"],
                    "activity":                activity_name,
                    "timestamp":               datetime.now().isoformat(),
                    "tracker_id":              msg["tracker_id"],
                    "absolute_bbox":           msg["bbox_data"],
                    "org_img":                 img_bytes,
                    "snap_shot":               img_bytes,
                    "video":                   None,
                    "triggered_by_hailo_event": metadata,
                    "imgsz":                   f"{w}:{h}",
                }
                try:
                    self.results_queue.put_nowait(full_message)
                except Exception as e:
                    logging.error(f"Queue Error: {e}")

        # ── Result callback → lets the activity update its own state ──── #
        if callable(on_result):
            try:
                on_result(
                    {
                        "missing_items": [
                            m.get("missing", []) for m in messages_to_send
                            if m["type"] == "VIOLATION_EVENT"
                        ],
                        "violation": bool(messages_to_send),
                        "tracker_id": tracker_id,
                    },
                    metadata,
                )
            except Exception as e:
                logging.error(f"on_result callback error: {e}")

    def _format_bbox(self, det, w, h):
        xmin, ymin, xmax, ymax = det['bbox']
        xywh_pct = [
            (xmin * 100) / w,
            (ymin * 100) / h,
            ((xmax - xmin) * 100) / w,
            ((ymax - ymin) * 100) / h,
        ]
        return [{
            "xywh":       xywh_pct,
            "class_name": det['class'],
            "subcategory": det['class'],
            "confidence": det['confidence'],
            "parameters": {},
            "anpr":       "False",
        }]
