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
        self.last_trigger_time = {}
        self.results_queue = None

    def set_results_queue(self, queue):
        self.results_queue = queue

    def should_trigger(self, activity_name):
        if not self.enabled: return False
        if activity_name not in self.target_activities: return False
        now = time.time()
        if (now - self.last_trigger_time.get(activity_name, 0)) < self.cooldown:
            return False
        return True

    def trigger(self, frame, activity_name, metadata=None):
        if not self.should_trigger(activity_name): return
        self.last_trigger_time[activity_name] = time.time()
        
        # Pass a COPY of the frame to the thread
        t = threading.Thread(target=self._send_request, args=(frame.copy(), activity_name, metadata))
        t.daemon = True
        t.start()

    def _send_request(self, frame, activity_name, metadata):
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(YOLOE_SERVER_URL, files=files, timeout=10.0)
            
            if response.status_code == 200:
                self._handle_result(response.json(), activity_name, metadata, frame)
            else:
                logging.error(f"YOLOE Server Error: {response.status_code}")
        except Exception as e:
            logging.error(f"YOLOE Request Failed: {e}")

    def _handle_result(self, result, activity_name, metadata, frame):
        detections = result.get("detections", [])
        detected_classes = [d['class'] for d in detections]
        
        rule = metadata.get('rule', 'report_all')
        required_items = metadata.get('required_items', [])
        tracker_id = metadata.get('hailo_tracker_id', 'unknown')
        
        messages_to_send = []
        
        # 1. ENCODE IMAGE ONCE
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            h, w = frame.shape[:2]
        except Exception as e:
            logging.error(f"Failed to encode image: {e}")
            return

        # Print what YOLOE found specifically for the required items
        if rule == 'must_detect':
            found_harness = False
            found_hook = False
            
            for det in detections:
                if det['class'] == 'harness':
                    found_harness = True
                    print(f"✅ DETECTED: Harness on Person {tracker_id} (Conf: {det['confidence']:.2f})")
                elif det['class'] == 'hook':
                    found_hook = True
                    print(f"✅ DETECTED: Hook on Person {tracker_id} (Conf: {det['confidence']:.2f})")

            # Check for missing items
            missing_items = [item for item in required_items if item not in detected_classes]
            
            if missing_items:
                # VIOLATION: At least one required item is missing
                print(f"⚠️ VIOLATION: Person {tracker_id} is missing: {', '.join(missing_items)}")
                
                # Get the person's bounding box passed from Hailo
                target_bbox = metadata.get('target_bbox', [0, 0, w, h]) 
                xmin, ymin, xmax, ymax = target_bbox
                
                # Calculate percentage for dashboard
                xywh_pct = [
                    (xmin * 100) / w,
                    (ymin * 100) / h,
                    ((xmax - xmin) * 100) / w,
                    ((ymax - ymin) * 100) / h
                ]

                # Format the violation message
                formatted_bbox = [{
                    "xywh": xywh_pct,
                    "class_name": "violation",
                    "subcategory": f"Missing {','.join(missing_items)}", 
                    "confidence": 1.0, 
                    "parameters": {},
                    "anpr": "False"
                }]
                
                messages_to_send.append({
                    "type": "VIOLATION_EVENT",
                    "tracker_id": str(tracker_id),
                    "bbox_data": formatted_bbox
                })

        # --- SCENARIO B: FORBIDDEN OBJECT CHECK ---
        elif rule == 'must_not_detect':
            for i, det in enumerate(detections):
                if det['class'] in required_items:
                    messages_to_send.append({
                        "type": "VIOLATION_EVENT",
                        "tracker_id": f"yoloe_{i}",
                        "bbox_data": self._format_bbox(det, w, h)
                    })

        # --- SCENARIO C: STANDARD REPORTING ---
        else:
            for i, det in enumerate(detections):
                messages_to_send.append({
                    "type": "YOLOE_DETECTION",
                    "tracker_id": f"yoloe_{i}",
                    "bbox_data": self._format_bbox(det, w, h)
                })

        # --- SEND MESSAGES TO KAFKA ---
        if self.results_queue:
            for msg in messages_to_send:
                full_message = {
                    "sensor_id": metadata.get("sensor_id", "Unknown"),
                    "type": msg["type"],
                    "activity": activity_name,
                    "timestamp": datetime.now().isoformat(),
                    "tracker_id": msg["tracker_id"],
                    "absolute_bbox": msg["bbox_data"],
                    "org_img": img_bytes,
                    "snap_shot": img_bytes,
                    "video": None,
                    "triggered_by_hailo_event": metadata,
                    "imgsz": f"{w}:{h}"
                }
                try:
                    self.results_queue.put_nowait(full_message)
                except Exception as e:
                    logging.error(f"Queue Error: {e}")

    def _format_bbox(self, det, w, h):
        """Helper to format a raw YOLOE bbox into the Dashboard format"""
        xmin, ymin, xmax, ymax = det['bbox']
        xywh_pct = [
            (xmin * 100) / w,
            (ymin * 100) / h,
            ((xmax - xmin) * 100) / w,
            ((ymax - ymin) * 100) / h
        ]
        return [{
            "xywh": xywh_pct,
            "class_name": det['class'],
            "subcategory": det['class'], 
            "confidence": det['confidence'],
            "parameters": {},
            "anpr": "False"
        }]
