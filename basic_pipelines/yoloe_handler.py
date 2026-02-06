import threading
import requests
import cv2
import time
import json
import logging
from datetime import datetime
import os

# --- CONFIGURATION ---
# This matches the port we set in Step 1
YOLOE_SERVER_URL = "http://localhost:5000/predict"

class YOLOEHandler:
    def __init__(self, config):
        """
        Initialize with the 'yoloe_control' section from configuration.json
        """
        self.config = config.get("yoloe_control", {})
        self.enabled = self.config.get("enabled", 0)
        
        # List of activities allowed to trigger YOLOE (e.g., ["UnauthorisedArea"])
        self.target_activities = set(self.config.get("activities", []))
        
        # Cooldown to prevent spamming the Docker container (in seconds)
        self.cooldown = self.config.get("cooldown_sec", 2)
        
        # Dictionary to track last trigger time per activity
        self.last_trigger_time = {}
        
        # Reference to the main Kafka/Event queue
        self.results_queue = None 

    def set_results_queue(self, queue):
        """
        Link the handler to the main event queue so we can push results
        to the dashboard/Kafka.
        """
        self.results_queue = queue

    def should_trigger(self, activity_name):
        """
        Check if we are allowed to run inference based on config and time.
        """
        if not self.enabled:
            return False
            
        if activity_name not in self.target_activities:
            return False

        now = time.time()
        last_time = self.last_trigger_time.get(activity_name, 0)
        
        # If we triggered this activity recently, skip it
        if (now - last_time) < self.cooldown:
            return False
            
        return True

    def trigger(self, frame, activity_name, metadata=None):
        """
        The main method called by detection.py.
        This runs ASYNCHRONOUSLY so it does NOT block the video pipeline.
        """
        if not self.should_trigger(activity_name):
            return

        # Update the cooldown timer immediately
        self.last_trigger_time[activity_name] = time.time()

        # Start a background thread to send the request
        t = threading.Thread(target=self._send_request, args=(frame.copy(), activity_name, metadata))
        t.daemon = True
        t.start()

    def _send_request(self, frame, activity_name, metadata):
        """
        Worker thread: Encodes image -> Sends to Docker -> Handles Result
        """
        try:
            # Encode frame to JPEG (required for HTTP transfer)
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            
            # Send POST request to the local Docker container
            # Timeout is important! If Docker hangs, don't let this thread hang forever.
            response = requests.post(YOLOE_SERVER_URL, files=files, timeout=2.0)
            
            if response.status_code == 200:
                result = response.json()
                self._handle_result(result, activity_name, metadata)
            else:
                logging.error(f"YOLOE Server Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            logging.error("YOLOE Connection Failed: Is the Docker container running?")
        except Exception as e:
            logging.error(f"YOLOE Request Failed: {e}")

    def _handle_result(self, result, activity_name, metadata):
        """
        Process the JSON result from YOLOE and push to the event queue.
        """
        detections = result.get("detections", [])
        
        # If nothing detected, do nothing (or you can log it)
        if not detections:
            return

        logging.info(f"YOLOE Validation Success: {activity_name} - Found {len(detections)} objects")

        # Create a standard message format for your Kafka/Dashboard
        if self.results_queue:
            for i, det in enumerate(detections):
                message = {
                    "sensor_id": metadata.get("sensor_id", "Unknown") if metadata else "Unknown",
                    "type": "YOLOE_DETECTION",
                    "activity": activity_name,
                    "timestamp": datetime.now().isoformat(),

                    # HARDCODED ID AS REQUESTED
                    "tracker_id": f"yoloe_{i}",

                    "class_name": det['class'],
                    "confidence": det['confidence'],
                    "absolute_bbox": det['bbox'],  # [xmin, ymin, xmax, ymax]

                    # Context from the trigger
                    "triggered_by_hailo_event": metadata
                }

                # Push to queue safely (Non-blocking)
                try:
                    self.results_queue.put_nowait(message)
                except Exception as e:
                    logging.error(f"Failed to push YOLOE result to queue: {e}")
