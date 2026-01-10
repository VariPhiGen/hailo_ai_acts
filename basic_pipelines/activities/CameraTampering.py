import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone,xywh_original_percentage
)
import numpy as np
import imagehash
from scipy.stats import entropy
import cv2

class CameraTampering:
    def __init__(self, parent,zone_data,parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent=parent
        self.relay=None

        self.parameters = parameters
        self.zone_data = zone_data
        self.running_data = {}

        #Initialize Relay
        if parameters["relay"]==1:
            try:
                if self.parent.relay_handler.device==None:
                    success = self.parent.relay_handler.initiate_relay()
                    if not success:
                        print("⚠️ Relay device not available. Continuing without relay control.")
                        self.relay = None
                        return
                self.relay=self.parent.relay_handler
                self.switch_relay=parameters["switch_relay"]
            except Exception as e:
                print(f"⚠️ Relay initialization failed: {e}. Continuing without relay control.")
                self.relay = None

        # Initiating Zone wise
        for zone_name in zone_data.keys():
            self.running_data[zone_name]={}
        
        self.last_check_time = self.parameters.get("last_check_time", 0)
        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        # Set up the timezone from the provided string
        self.timezone = pytz.timezone(self.timezone_str)

        #Extra Variable Related to Activity
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=False
        )
        self.tampering_check=0
        self.prev_brightness=None
        self.ref_hash=None

    def run(self):
        frame=self.parent.model_image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        variance = np.var(gray)
        
        # Histogram-based entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        entropy_val = entropy(hist_norm + 1e-6)  # Add epsilon to avoid log(0)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size

        #print(f"Brightness: {brightness:.2f}, Var: {variance:.2f}, Entropy: {entropy_val:.2f}, Edges: {edge_ratio:.4f}")

        # Initialize brightness tracking
        if self.prev_brightness is None:
            self.prev_brightness = brightness

        brightness_change = abs(brightness - self.prev_brightness)
        self.prev_brightness = brightness

        tampering = (
            brightness_change > self.parameters["brightness_thres"] or
            variance < 15 or
            entropy_val < 1.5 or
            edge_ratio < 0.008
        )

        # Initialize ref_hash if it has not been set (first-time setup or reset)
        if self.ref_hash is None:
            self.ref_hash = self.frame_hash(frame)

        # Compare current frame to reference frame for camera movement detection
        if self.ref_hash is not None:
            current_hash = self.frame_hash(frame)
            hash_diff = current_hash - self.ref_hash
        #print(self.ref_hash)
            

        # Check tampering condition based on time intervals and event triggering
        if int(time.time() - self.running_data["alert_interval"]) > self.running_data["alert_interval"]:
            if tampering or hash_diff > 20:
                self.running_data["frame_count"] += 1

            if self.running_data["frame_count"] > 0:
                xywh = [0, 0, 100, 100]
                datetimestamp = f"{datetime.now(self.timezone).isoformat()}"
                if hash_diff >20:
                    self.create_result_events(xywh, "Tampering", "Security-Camera_Movement", {}, datetimestamp, 1,self.parent.image)
                    print("Work Done Movement")
                else:    
                    self.create_result_events(xywh, "Tampering", "Security-Camera_Tampering", {}, datetimestamp, 1,self.parent.image)
                    print("Work Done Tampering")

                # Update the reference frame after tampering
                self.ref_hash = current_hash  # Update reference frame to the current one

                # Reset tampering data
                self.running_data["frame_count"] = 0
                self.running_data["alert_interval"] = time.time()
                self.prev_brightness = None

    
    def frame_hash(self,image):
        """Generate perceptual hash of the image to compare frames."""
        return imagehash.phash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        
