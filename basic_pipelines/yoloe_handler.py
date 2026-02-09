import threading
import requests
import cv2
import time
import json
import logging
from datetime import datetime
import os

# --- CONFIGURATION ---
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
        """Link to the main event queue"""
        self.results_queue = queue

    def should_trigger(self, activity_name):
        if not self.enabled: 
            return False
        if activity_name not in self.target_activities: 
            return False
        
        now = time.time()
        last_time = self.last_trigger_time.get(activity_name, 0)
        
        if (now - last_time) < self.cooldown:
            return False
        return True

    def trigger(self, frame, activity_name, metadata=None):
        if not self.should_trigger(activity_name):
            return

        self.last_trigger_time[activity_name] = time.time()

        # Start background thread
        # Pass frame.copy() to ensure thread safety
        t = threading.Thread(target=self._send_request, args=(frame.copy(), activity_name, metadata))
        t.daemon = True
        t.start()

    def _send_request(self, frame, activity_name, metadata):
        """
        Worker thread: Sends image to Docker -> Handles Result
        """
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            
            response = requests.post(YOLOE_SERVER_URL, files=files, timeout=2.0)
            
            if response.status_code == 200:
                result = response.json()
                # PASS FRAME to handle_result
                self._handle_result(result, activity_name, metadata, frame)
            else:
                logging.error(f"YOLOE Server Error: {response.status_code}")
                
        except Exception as e:
            logging.error(f"YOLOE Request Failed: {e}")

    def _handle_result(self, result, activity_name, metadata, frame):
        """
        Process results and push to Kafka queue.
        """
        detections = result.get("detections", [])
        
        if not detections:
            return

        # 1. ENCODE IMAGE TO BYTES (Crucial)
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
        except Exception as e:
            logging.error(f"Failed to encode image for queue: {e}")
            return

        # Get image dimensions
        h, w = frame.shape[:2]

        if self.results_queue:
            for i, det in enumerate(detections):
                
                # --- NEW PRINT LOGIC --------------------------------------
                # Prints details for any detection with high confidence
                if det['confidence'] > 0.2:
                    print(f"🎯 YOLOE DETECTED: {det['class']} | Confidence: {det['confidence']:.2f}")
                # ----------------------------------------------------------

                # 2. Calculate XYWH in Percentages
                xmin, ymin, xmax, ymax = det['bbox']
                box_w = xmax - xmin
                box_h = ymax - ymin
                
                xywh_pct = [
                    (xmin * 100) / w,
                    (ymin * 100) / h,
                    (box_w * 100) / w,
                    (box_h * 100) / h
                ]

                # 3. FORMAT STRUCTURE
                formatted_bbox = [{
                    "xywh": xywh_pct,
                    "class_name": det['class'],
                    "subcategory": det['class'], 
                    "confidence": det['confidence'],
                    "parameters": {},
                    "anpr": "False"
                }]

                message = {
                    "sensor_id": metadata.get("sensor_id", "Unknown") if metadata else "Unknown",
                    "type": "YOLOE_DETECTION",
                    "activity": activity_name,
                    "timestamp": datetime.now().isoformat(),
                    "tracker_id": f"yoloe_{i}", 
                    "absolute_bbox": formatted_bbox,
                    "org_img": img_bytes,
                    "snap_shot": img_bytes, 
                    "video": None,
                    "triggered_by_hailo_event": metadata,
                    "imgsz": f"{w}:{h}"
                }
                
                try:
                    self.results_queue.put_nowait(message)
                except Exception as e:
                    logging.error(f"Failed to push YOLOE result to queue: {e}")
