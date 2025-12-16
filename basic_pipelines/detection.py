from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import sys
import numpy as np
import cv2
import hailo
import concurrent.futures
from pathlib import Path
import importlib
from dotenv import load_dotenv

# External Libraries
import json
from collections import defaultdict, deque
import queue
from shapely.geometry import Point, Polygon
import pytz
from threading import Thread, Lock
from datetime import datetime
import time
import asyncio
import argparse
import atexit
import signal
from snapshotapi import Snapshot

# Local modules
from kafka_handler import KafkaHandler
from radar_handler import RadarHandler
from relay_handler import Relay
from helper_utils import (
    setup_logging, encode_frame_to_bytes, is_vehicle_in_zone, crop_image_numpy,
    closest_line_projected_distance, get_unique_tracker_ids, calculate_distance
)


from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp



# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        # Optimized thread pool for I/O operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=6,  # More workers for parallel I/O
            thread_name_prefix="detection_worker"
        )
        
        # Camera Variables
        self.sensor_id = None
        self.hef_path=None
        self.labels_json=None
        self.zone_data = None
        self.violation_id_data = None
        self.parameters_data = None
        
        # Image Variables
        self.model_image=None
        self.image = None
        self.original_height = None
        self.original_width = None
        self.ratio = None
        self.padx = None
        self.pady = None
        self.detection_boxes = None
        self.classes = None
        self.tracker_ids = None
        self.anchor_points_original = None
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.last_n_frame_tracker_ids = None
        # Initialize a deque to store tracker_ids for the last n frames
        self.LNFCTI = deque(maxlen=30)
        self.time_stamp = deque(maxlen=4)
        self.cleaning_time_for_events = time.time()
        
        # Frame monitoring (for external monitoring only)
        self.frame_monitor_count = 0

        #Relay handler
        self.relay_handler=Relay()

        # Radar handler
        self.radar_handler = RadarHandler()
        self.radar_maxdiff = None
        self.save_snapshots = None
        self.save_rtsp_images =None
        self.take_video=None
        self.ccai=time.time()

        # Traffic Overspeeding Distancewise Variables
        self.traffic_overspeeding_distancewise_data = None
        self.distancewise_tracking = None
        self.calibrate_class_wise = {}
        self.calibrate_speed = {}
        self.calibrated_classes = set()  # Track which classes are done calibrating

        # Queue for sending Data to Kafka
        self.results_analytics_queue = None
        self.results_events_queue = None
        
        # Active methods
        self.active_methods = None
        self.active_activities_for_cleaning = {}
        
        # Run Triggering Loop for raving Camera Snapshot
        self.cam = None
        self.api_mode=None
        self.kafka_mode=None

        # Kafka Handler
        self.kafka_handler=None

        #recorder
        self.recorder=None
        
        # Initialize cropping directory path once
        self.cropping_dir = os.path.join(os.getcwd(), "SVDS2", "original_croppings")
        # Create directory if it doesn't exist
        os.makedirs(self.cropping_dir, exist_ok=True)
        
        self.main_loop = asyncio.new_event_loop()
        self.asyncio_thread = Thread(target=self.start_asyncio_loop, daemon=True)
        self.asyncio_thread.start()
        
    def calibration_check(self,flag=False):
        """
        Compute a per-class calibration factor (radar_speed / ai_speed),
        store it in self.calibrate_class_wise and then clear calibration inputs.
        """
        cs = self.calibrate_speed
        ai_speed = cs.get("speed")
        radar_speed = cs.get("radar")
        cls = cs.get("class_name")

        # Validate required inputs quickly
        try:
            ai = float(ai_speed)
            rs = float(radar_speed)
        except (TypeError, ValueError):
            return  # invalid or missing data—nothing to do

        # Compute and store calibration factor
        if (flag is True and abs(ai-rs)<self.radar_maxdiff) or flag is False:
            self.calibrate_class_wise[cls] = rs*self.calibrate_class_wise[cls]/ ai
        elif (time.time()-self.ccai) > 300:
            self.calibrate_class_wise[cls] = rs*self.calibrate_class_wise[cls]/ ai
        # Reset input values efficiently
        # Use assignment to None
        cs["speed"] = cs["radar"] = cs["class_name"] = None

        return True
        
    def start_asyncio_loop(self):
        asyncio.set_event_loop(self.main_loop)
        self.main_loop.run_forever()
        
    async def trigger_snapshot_loop(self, xywh, class_name,subcategory, parameters, datetimestamp, confidence=1, image=None,anpr_status=None):
        # Use existing main_loop instead of creating new one
        try:
            suffix = f"{class_name}_{subcategory}"
            
            # Keep camera capture as-is (temporary files)
            filename, cgi_snapshot = None, None
            if hasattr(self, 'cam') and self.cam:
                filename, cgi_snapshot = self.cam.capture(prefix=str(suffix))
            
            if hasattr(self, 'save_snapshots') and filename and cgi_snapshot:
                # Keep file saving as-is (temporary files)
                await self.main_loop.run_in_executor(
                    self.thread_pool,
                    self._save_snapshot,
                    filename, cgi_snapshot
                )
        except Exception as e:
            print(f"DEBUG: Error in snapshot capture: {e}")
        
        # Generate video bytes first (with timeout)
        video_bytes = None
        if hasattr(self, 'take_video') and self.recorder is not None:
            try:
                video_bytes = await asyncio.wait_for(
                    self.main_loop.run_in_executor(
                        self.thread_pool,
                        self.recorder.generate_video_bytes
                    ),
                    timeout=10.0  # 10-second timeout
                )
            except asyncio.TimeoutError:
                pass  # Continue without video if timeout
            except Exception as e:
                print(f"DEBUG: Video generation error: {e}")
            
        # Lightweight image processing (immediate)
        final_image, height, width, anpr_status = self._process_image_lightweight(image,anpr_status)

        
        # Fire-and-forget RTSP save
        if hasattr(self, 'save_rtsp_images') and self.save_rtsp_images and self.recorder is not None:
            asyncio.create_task(self._async_save_rtsp(image, suffix))
        
        # Create message with complete data
        message = {
            "sensor_id": self.sensor_id,
            "org_img": final_image,
            "snap_shot": cgi_snapshot,
            "video": video_bytes,  # Video bytes ready before queuing
            "absolute_bbox": [{"xywh": xywh,"subcategory":subcategory, "class_name": class_name, "confidence": confidence, "parameters": parameters, "anpr": anpr_status}],
            "datetimestamp": datetimestamp,
            "imgsz": f"{width}:{height}",
            "color": "#FFFF00"
        }
        
        #API Calling

        # Queue message with complete data
        if user_data.api_mode==1 or user_data.kafka_mode==1:
            self._queue_message_non_blocking(message)

    def _save_snapshot(self, filename, cgi_snapshot):
        """Synchronous snapshot saving in thread pool."""
        SAVE_FOLDER = "./snapshots"
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        path = os.path.join(SAVE_FOLDER, filename)
        with open(path, "wb") as f:
            f.write(cgi_snapshot)
    

    
    def _process_image_lightweight(self, image,anpr_status):
        """Lightweight image processing without heavy operations."""
        try:
            if anpr_status is not None:
                # Minimal processing - just get dimensions
                anprimage_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = encode_frame_to_bytes(anprimage_rgb, 100)
                height, width = image.shape[:2]
                anpr_status = "True"
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                image = encode_frame_to_bytes(image_rgb, 90)  # Lower quality for speed
                anpr_status = "False"
            return image, height, width, anpr_status
        except Exception as e:
            print(f"DEBUG: Error in image processing: {e}")

    
    async def _async_save_rtsp(self, image, suffix):
        """Fire-and-forget RTSP image save."""
        try:
            img_for_rtsp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else self.image
            await self.main_loop.run_in_executor(
                self.thread_pool,
                self.recorder.save_images,
                img_for_rtsp, self.cropping_dir, suffix
            )
        except Exception as e:
            print(f"DEBUG: Error in RTSP save: {e}")
    
    def _queue_message_non_blocking(self, message):
        """Drop old messages to make space for new ones when queue is full."""
        try:
            queue_size = self.results_events_queue.qsize()
            
            # If queue is getting full, drop old messages to make space
            if queue_size >= 80:
                try:
                    # Drop oldest message to make space for new one
                    old_message = self.results_events_queue.get_nowait()
                except queue.Empty:
                    pass  # Queue was cleared by another thread
                
                # Now add the new message
                self.results_events_queue.put_nowait(message)
                return
            
            # Normal case: add message if queue has space
            self.results_events_queue.put_nowait(message)
            
        except queue.Full:
            # Last resort: force drop old message and add new one
            try:
                old_message = self.results_events_queue.get_nowait()
                
                
                # Add new message
                self.results_events_queue.put_nowait(message)

                
            except Exception as e:
                print(f"❌ Failed to replace message: {e}")
                # Silently drop if all else fails

# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
# -----------------------------------------------------------------------------------------------

    def create_result_analytics(self, timestamp, analytics_name, class_name, zone_name, type, value):
        message = {
            "sensor_id": self.sensor_id,
            "datetimestamp": timestamp,
            "analytic_type": analytics_name,
            "class_name": class_name,
            "area": zone_name,
            "type": type,
            "value": value
        }
        try:
            self.results_analytics_queue.put_nowait(message)
        except queue.Full:
            # Handle gracefully without blocking
            pass

    def create_result_events(self, xywh, class_name,subcategory, parameters, datetimestamp, confidence=1, image=None,anpr_status=None):
        asyncio.run_coroutine_threadsafe(self.trigger_snapshot_loop(xywh, class_name,subcategory, parameters, datetimestamp, confidence,image,anpr_status), self.main_loop)

    def cleaning_events_data_with_last_frames(self):
        # Cleaning Violations
        for activity,method in self.active_activities_for_cleaning.items():
            if activity == "traffic_overspeeding_distancewise":
                for tracker_id in list(self.traffic_overspeeding_distancewise_data.keys()):
                    # Check if tracker_id is not in self.last_n_frame_tracker_ids
                    if tracker_id not in self.last_n_frame_tracker_ids:
                        # Remove the tracker_id from self.traffic_overspeeding_distancewise_data
                        del self.traffic_overspeeding_distancewise_data[tracker_id]
            else:
                method()
    
    # Activities Logics
    def traffic_overspeeding_distancewise(self):
        overspeeding_result = []
        vehicle_index = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["traffic_overspeeding_distancewise"]["speed_limit"].keys()]
        count = 0
        image=None
        
        # �� MOST CRITICAL: Add this early exit
        if not vehicle_index:
            return 
        
        for idx in vehicle_index:
            box = self.detection_boxes[idx]
            obj_class = self.classes[idx]
            tracker_id = self.tracker_ids[idx]
            anchor = self.anchor_points_original[idx]
            if tracker_id in self.violation_id_data["traffic_overspeeding_distancewise"]:
                continue
            for zone_name, zone_polygon in self.zone_data["traffic_overspeeding_distancewise"].items():
                if is_vehicle_in_zone(anchor, zone_polygon):
                    count += 1
                    if tracker_id not in self.distancewise_tracking and tracker_id not in self.violation_id_data["traffic_overspeeding_distancewise"]:
                        self.distancewise_tracking.append(tracker_id)
                        self.traffic_overspeeding_distancewise_data[tracker_id]["entry_anchor"] = anchor
                        self.traffic_overspeeding_distancewise_data[tracker_id]["entry_time"] = self.time_stamp[-1]
                        
                elif tracker_id in self.distancewise_tracking and tracker_id not in self.violation_id_data["traffic_overspeeding_distancewise"]:
                    vehicle_path = [anchor, self.traffic_overspeeding_distancewise_data[tracker_id]["entry_anchor"]]
                    self.distancewise_tracking.remove(tracker_id)
                    if vehicle_path[0][1] > vehicle_path[1][1]:
                        projected_distance, total_distance, lane_name = closest_line_projected_distance(vehicle_path, self.parameters_data["traffic_overspeeding_distancewise"]["lines"])
                        if projected_distance > (0.3*self.parameters_data["traffic_overspeeding_distancewise"]["lines_length"][lane_name]):
                            distance = float(self.parameters_data["traffic_overspeeding_distancewise"]["real_distance"] * projected_distance / self.parameters_data["traffic_overspeeding_distancewise"]["lines_length"][lane_name])
                            
                            speed = float(int(distance * 3.6 / (self.time_stamp[-1] - self.traffic_overspeeding_distancewise_data[tracker_id]["entry_time"])*float(self.calibrate_class_wise[obj_class])))
                            try:
                                radar_speed,rank1 = self.radar_handler.get_radar_data(speed, self.parameters_data["traffic_overspeeding_distancewise"]["speed_limit"][obj_class],obj_class)
                            except Exception as e:
                                print(f"CRITICAL: Radar data retrieval error: {e}")
                                cleanup_resources()
                                sys.exit(0)
                            
                            if radar_speed is not None and speed != 0 and rank1 and radar_speed !=0:
                                self.calibrate_speed["speed"] = speed
                                self.calibrate_speed["class_name"] = obj_class
                                self.calibrate_speed["radar"] = radar_speed
                                if obj_class not in self.calibrated_classes and self.calibration_check() and self.radar_handler.is_calibrating[obj_class] is False:
                                    self.calibrated_classes.add(obj_class)
                                else:
                                    self.calibration_check(True)
                            if obj_class in self.calibrated_classes:
                                overspeeding_result.append({"tracker_id": tracker_id, "box": box, "speed": speed, "radar_speed": radar_speed, "lane_name": lane_name, "obj_class": obj_class})
        
        for result in overspeeding_result:
            obj_class = result["obj_class"]
            if result["radar_speed"] is not None and 120 >= result["radar_speed"] > (self.parameters_data["traffic_overspeeding_distancewise"]["speed_limit"][result["obj_class"]]):
                # Get the bounding rectangle of the polygon
                self.violation_id_data["traffic_overspeeding_distancewise"].append(result["tracker_id"])
                anprimage = crop_image_numpy(self.image, result["box"])
                xywh = [0, 0, 100, 100]
                datetimestamp = f"{datetime.now(self.ist_timezone).isoformat()}"
                self.create_result_events(xywh, obj_class,"Traffic Ssafety-Overspeeding", {"speed": result["radar_speed"],"tag":"RDR"}, datetimestamp, 1,anprimage,True)
            
            elif 55 > result["speed"] > (self.parameters_data["traffic_overspeeding_distancewise"]["speed_limit"][result["obj_class"]])+5 and result["radar_speed"] is None:
                # Get the bounding rectangle of the polygon
                self.violation_id_data["traffic_overspeeding_distancewise"].append(result["tracker_id"])
                anprimage = crop_image_numpy(self.image, result["box"])
                xywh = [0, 0, 100, 100]
                datetimestamp = f"{datetime.now(self.ist_timezone).isoformat()}"
                self.create_result_events(xywh, obj_class,"Traffic Ssafety-Overspeeding", {"speed": result["speed"],"tag":"AI"}, datetimestamp, 1, anprimage,True)
# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data,frame_type):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Simple frame counter for external monitoring
    user_data.frame_monitor_count += 1
    
    # Reset counter if it reaches 1000 to prevent overflow
    if user_data.frame_monitor_count >= 1000:
        user_data.frame_monitor_count = 0
        #print("INFO: Frame monitor counter reset to prevent overflow")

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    
    if frame_type == "original":
        # print(user_data.original_frame_count)
        if format is not None and width is not None and height is not None:
            # Get video frame
            user_data.image = get_numpy_from_buffer(buffer, format, width, height)
            # user_data.image = cv2.cvtColor(user_data.image, cv2.COLOR_BGR2RGB)
            if user_data.original_height is None:
                user_data.original_height=height
                user_data.original_width=width
    else:
        # If the user_data.use_frame is set to True, we can get the video frame from the buffer
        frame = None
        if format is not None and width is not None and height is not None:
            buf_timestamp = buffer.pts  # nanoseconds
            ts = buf_timestamp / Gst.SECOND
            #print(ts)
            user_data.time_stamp.append(ts)
            # Get video frame
            frame = get_numpy_from_buffer(buffer, format, width, height)
            user_data.model_image=frame
            if user_data.recorder is not None:
                user_data.recorder.add_frame(frame)
            
            
        # Get the detections from the buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        #user_data.time_stamp.append(tr)

        # Parse the detections
        detection_count = 0
        xyxys=[]
        class_ids=[]
        class_names=[]
        confidences=[]
        tracker_ids=[]
        anchor_points_original=[]
        
        if user_data.ratio is None and user_data.original_width is not None:
            user_data.ratio=min(width/user_data.original_width,height/user_data.original_height)
            user_data.padx=int((width - user_data.original_width*user_data.ratio)/2)
            user_data.pady=int((height - user_data.original_height*user_data.ratio)/2)
        else:
            for detection in detections:
                label = detection.get_label()
                bbox = detection.get_bbox()
                confidence = detection.get_confidence()
                # Get track ID
                track_id = 0
                track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                if len(track) == 1:
                    track_id = track[0].get_id()
                    tracker_ids.append(track_id)
                    x_min=max(int((bbox.xmin()*width-user_data.padx)/user_data.ratio),0)
                    y_min=max(int((bbox.ymin()*height-user_data.pady)/user_data.ratio),0)
                    x_max=max(int((bbox.xmax()*width-user_data.padx)/user_data.ratio),0)
                    y_max=max(int((bbox.ymax()*height-user_data.pady)/user_data.ratio),0)
                    #Appending the results from Detections
                    xyxys.append([x_min,y_min,x_max,y_max])
                    class_ids.append(detection.get_class_id())
                    class_names.append(label)
                    confidences.append(confidence)
                    anchor_points_original.append((((x_min+x_max)/ 2),(y_max)))
                
                detection_count += 1
            
            # Updating the Activities Instance
            user_data.detection_boxes=xyxys
            user_data.classes= class_names
            user_data.tracker_ids=tracker_ids
            user_data.anchor_points_original=anchor_points_original
            
            # Last n frame tracker ids
            user_data.LNFCTI.append(tracker_ids)
            user_data.last_n_frame_tracker_ids = get_unique_tracker_ids(user_data.LNFCTI)

            # BEFORE: Clean every 60 seconds
            if int((time.time()-user_data.cleaning_time_for_events)) > 120:
                user_data.cleaning_events_data_with_last_frames() 
                user_data.cleaning_time_for_events=time.time()

            for method in user_data.active_methods:
                method()

    return Gst.PadProbeReturn.OK

def cleanup_resources():
    """Clean up resources before exiting."""
    
    try:
        # Close Kafka handler
        if 'kafka_handler' in globals():
            print("📡 Closing Kafka handler...")
            kafka_handler.close()
        
        # Stop radar if running
        if 'user_data' in globals() and hasattr(user_data, 'radar_handler'):
            print("📡 Stopping radar handler...")
            user_data.radar_handler.stop_radar()
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def load_activity_class(activity_name):
    """Load activity class dynamically from basic_pipelines/activities/"""
    try:
        module = importlib.import_module(f"basic_pipelines.activities.{activity_name}")
        return getattr(module, activity_name)  # Class name must match filename
    except Exception as e:
        print(f"❌ Failed to load {activity_name}: {e}")
        return None
    
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # load .env into process environment
    load_dotenv(env_path_str)
    # Logging is disabled - all errors sent to Kafka
    setup_logging()

    # Register cleanup on exit
    atexit.register(cleanup_resources)
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\n⚠️  Received interrupt signal, cleaning up...")
        cleanup_resources()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create an instance of the user app callback class
    user_data = user_app_callback_class()

    # Load the JSON data from the file
    try:
        with open('configuration.json', 'r') as file:
            config = json.load(file)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration: {e}")
        cleanup_resources()
        sys.exit(0)
    
    # === Overlay sensitive values from .env into config (so KafkaHandler etc read them) ===
    kafka_vars = config.get("kafka_variables", {})
    # If BROKER envs present, override
    if os.getenv("BROKER_PRIMARY"):
        kafka_vars["primary_broker"] = os.getenv("BROKER_PRIMARY")
    if os.getenv("BROKER_SECONDARY"):
        kafka_vars["secondary_broker"] = os.getenv("BROKER_SECONDARY")
    if os.getenv("BROKER_PRIMARY") or os.getenv("BROKER_SECONDARY"):
        kafka_vars["broker_failover_timeout"]= int(os.getenv("BROKER_FAILOVER_TIMEOUT", 30)),
        kafka_vars["send_analytics_pipeline"]= os.getenv("SEND_ANALYTICS_PIPELINE", "variphianalytics"),
        kafka_vars["send_events_pipeline"]= os.getenv("SEND_EVENTS_PIPELINE", "variphievents"),
        kafka_vars["log_topic"]= os.getenv("LOG_TOPIC", "error_log")

    # AWS S3 overrides (primary / secondary)
    s3 = kafka_vars.get("AWS_S3", {})
    if os.getenv("AWS_PRIMARY_KEY") and os.getenv("AWS_PRIMARY_SECRET"):
        s3.setdefault("primary", {})
        s3["primary"]["aws_access_key_id"] = os.getenv("AWS_PRIMARY_KEY")
        s3["primary"]["aws_secret_access_key"] = os.getenv("AWS_PRIMARY_SECRET")
        s3["primary"]["end_point_url"] = os.getenv("AWS_PRIMARY_ENDPOINT")
        s3["primary"]["region_name"] = os.getenv("AWS_PRIMARY_REGION", "ap-south-1")
        s3["primary"]["BUCKET_NAME"] = os.getenv("AWS_PRIMARY_BUCKET_NAME", "arrestovideos")
        s3["primary"]["org_img_fn"] = os.getenv("AWS_PRIMARY_ORG_IMG_FN", "violationoriginalimages/")
        s3["primary"]["video_fn"] = os.getenv("AWS_PRIMARY_VIDEO_FN", "videoclips/")
        s3["primary"]["cgi_fn"] = os.getenv("AWS_PRIMARY_CGI_FN", "cgisnapshots/")
   
    if os.getenv("AWS_SECONDARY_KEY") and os.getenv("AWS_SECONDARY_SECRET"):
        s3.setdefault("secondary", {})
        s3["secondary"]["aws_access_key_id"] = os.getenv("AWS_SECONDARY_KEY")
        s3["secondary"]["aws_secret_access_key"] = os.getenv("AWS_SECONDARY_SECRET")
        s3["secondary"]["end_point_url"] = os.getenv("AWS_SECONDARY_ENDPOINT")
        s3["secondary"]["region_name"] = os.getenv("AWS_SECONDARY_REGION", "ap-south-1")
        s3["secondary"]["BUCKET_NAME"] = os.getenv("AWS_SECONDARY_BUCKET_NAME", "arrestovideos")
        s3["secondary"]["org_img_fn"] = os.getenv("AWS_SECONDARY_ORG_IMG_FN", "violationoriginalimages/")
        s3["secondary"]["video_fn"] = os.getenv("AWS_SECONDARY_VIDEO_FN", "videoclips/")
        s3["secondary"]["cgi_fn"] = os.getenv("AWS_SECONDARY_CGI_FN", "cgisnapshots/")

    if os.getenv("API_AWS_PRIMARY_KEY") and os.getenv("API_AWS_PRIMARY_SECRET"):
        s3.setdefault("api_primary", {})
        s3["api_primary"]["aws_access_key_id"] = os.getenv("API_AWS_PRIMARY_KEY")
        s3["api_primary"]["aws_secret_access_key"] = os.getenv("API_AWS_PRIMARY_SECRET")
        s3["api_primary"]["end_point_url"] = os.getenv("API_AWS_PRIMARY_ENDPOINT")
        s3["api_primary"]["region_name"] = os.getenv("API_AWS_PRIMARY_REGION", "ap-south-1")
        s3["api_primary"]["BUCKET_NAME"] = os.getenv("API_AWS_PRIMARY_BUCKET_NAME", "arrestovideos")
        s3["api_primary"]["org_img_fn"] = os.getenv("API_AWS_PRIMARY_ORG_IMG_FN", "violationoriginalimages/")
        s3["api_primary"]["video_fn"] = os.getenv("API_AWS_PRIMARY_VIDEO_FN", "videoclips/")
        s3["api_primary"]["cgi_fn"] = os.getenv("API_AWS_PRIMARY_CGI_FN", "cgisnapshots/")

    if os.getenv("API_AWS_SECONDARY_KEY") and os.getenv("API_AWS_SECONDARY_SECRET"):
        s3.setdefault("api_secondary", {})
        s3["api_secondary"]["aws_access_key_id"] = os.getenv("API_AWS_SECONDARY_KEY")
        s3["api_secondary"]["aws_secret_access_key"] = os.getenv("API_AWS_SECONDARY_SECRET")
        s3["api_secondary"]["end_point_url"] = os.getenv("API_AWS_SECONDARY_ENDPOINT")
        s3["api_secondary"]["region_name"] = os.getenv("API_AWS_SECONDARY_REGION", "ap-south-1")
        s3["api_secondary"]["BUCKET_NAME"] = os.getenv("API_AWS_SECONDARY_BUCKET_NAME", "arrestovideos")
        s3["api_secondary"]["org_img_fn"] = os.getenv("API_AWS_SECONDARY_ORG_IMG_FN", "violationoriginalimages/")
        s3["api_secondary"]["video_fn"] = os.getenv("API_AWS_SECONDARY_VIDEO_FN", "videoclips/")
        s3["api_secondary"]["cgi_fn"] = os.getenv("API_AWS_SECONDARY_CGI_FN", "cgisnapshots/")
    
    s3["s3_failover_timeout"]= int(os.getenv("S3_FAILOVER_TIMEOUT", 30)),
    s3["upload_retries"]= int(os.getenv("S3_UPLOAD_RETRIES", 3))

    kafka_vars["AWS_S3"] = s3
    config["kafka_variables"] = kafka_vars

    # Setup the Hef-path with path existence check
    try:
        hef_path = config.get("default_arguments", {}).get("hef_path")
        labels_json = config.get("default_arguments", {}).get("labels-json")
        
        # Check if hef_path exists and is not None/empty
        if hef_path and hef_path != "None" and os.path.exists(hef_path):
            user_data.hef_path = hef_path
        else:
            user_data.hef_path = None
            print(f"⚠️  HEF file not found or invalid: {hef_path}")
        
        # Check if labels_json exists and is not None/empty
        if labels_json and labels_json != "None" and os.path.exists(labels_json):
            user_data.labels_json = labels_json
        else:
            user_data.labels_json = None
            print(f"⚠️  Labels file not found or invalid: {labels_json}")
            
    except Exception as e:
        print(f"❌ Error setting up file paths: {e}")
        user_data.hef_path = None
        user_data.labels_json = None
    

    # Get save settings from config
    save_settings = config.get("save_settings", {})
    user_data.save_snapshots = bool(save_settings.get("save_snapshots", 0))
    user_data.save_rtsp_images = bool(save_settings.get("save_rtsp_images", 0))
    user_data.take_video=bool(save_settings.get("take_video", 0))
    print(f"Save settings loaded from config: snapshots={user_data.save_snapshots}, rtsp_images={user_data.save_rtsp_images}")
   
        
    # Snapshot if configured
    if "camera_details" in config:
        try:
            cam_config = config["camera_details"]
            if bool(save_settings.get("take_cgi_snapshots", 0)):
                user_data.cam = Snapshot(
                    camera_ip=cam_config.get("camera_ip"),
                    user=cam_config.get("username"),
                    pwd=cam_config.get("password")
                )
        except Exception as e:
            print("DEBUG: Failed to initialize CGI Snapshots: {e}")
            cleanup_resources()
            sys.exit(0)
    
    
    # Initialize results queue
    results_analytics_queue = queue.Queue(maxsize=1)
    results_events_queue = queue.Queue(maxsize=100)

    # Getting Kafka Configuration
    kafka_variables = config.get("kafka_variables", {})
    dashboard_contectivity=config.get("dashboard_connectivity",{})
    user_data.api_mode=dashboard_contectivity.get("api",0)
    user_data.kafka_mode=dashboard_contectivity.get("kafka",0)

    # Creating Thread for Kafka
    if user_data.api_mode==1 or user_data.kafka_mode==1:
        # Set up Kafka error logging for radar handler
        try:
            print(config)
            kafka_handler = KafkaHandler(config)
            user_data.kafka_handler = kafka_handler
            # Add recorder to user_data for frame recording
            user_data.recorder = kafka_handler.recorder
            
        except Exception as e:
            print(f"CRITICAL: Failed to initialize Kafka handler: {e}")
        kafka_thread = Thread(target=kafka_handler.run_kafka_loop, args=(results_events_queue, results_analytics_queue,user_data.api_mode,user_data.kafka_mode))
        kafka_thread.daemon = True
        kafka_thread.start()
    
    #Getting Data available
    sensor_id=config.get("sensor_id")
    available_activities = config.get("available_activities")
    active_activities = config.get("active_activities")
    user_data.sensor_id = sensor_id
    user_data.results_analytics_queue = results_analytics_queue
    user_data.results_events_queue = results_events_queue
    
    # Extract zones data for each activity in activities_data
    zones_data = {}
    parameters_data={}
    violation_id_data={}

    active_instances=[]
    active_methods=[]
    for activity, details in config["activities_data"].items():
        if activity in available_activities and activity in active_activities:
            if activity=="traffic_overspeeding_distancewise":
                # Initialize radar if configured
                if "radar_config" in config:
                    try:
                        radar_config = config["radar_config"]
                        user_data.radar_maxdiff=radar_config.get("max_diff_rais", 15)
                        user_data.radar_handler.init_radar(
                            port=radar_config.get("port", "/dev/ttyACM0"),
                            baudrate=radar_config.get("baudrate", 9600),
                            max_age=radar_config.get("max_age", 10),
                            max_diff_rais=radar_config.get("max_diff_rais", 15),
                            calibration_required=config.get("calibration_required", 2)
                        )
                        user_data.radar_handler.start_radar()
                    except Exception as e:
                        print(f"CRITICAL: Failed to initialize radar: {e}")
                        continue
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]
                user_data.traffic_overspeeding_distancewise_data=defaultdict(lambda: defaultdict(dict))
                user_data.distancewise_tracking=[]
                parameters_data["traffic_overspeeding_distancewise"]["lines_length"]={}
                user_data.time_stamp=deque(maxlen=20)
                # Initialize calibration count to 0 for each class
                for class_name in details["parameters"]["speed_limit"].keys():
                    user_data.radar_handler.class_calibration_count[class_name] = 0
                    user_data.radar_handler.is_calibrating[class_name]=True
                
                for class_name, limit in details["parameters"]["speed_limit"].items():
                    # Calibrate using your logic; here's an example:
                    user_data.calibrate_class_wise[class_name] = details["parameters"]["calibration"]
                user_data.calibrate_speed["speed"] = None
                user_data.calibrate_speed["class_name"] = None
                user_data.calibrate_speed["radar"] = None
                user_data.calibrate_speed["last_seen"] = time.time() 
                    
                for lane_name,coordinates in parameters_data[activity]["lines"].items():
                    total_distance = 0
                    # Iterate over the pairs of coordinates in the line
                    for i in range(len(coordinates) - 1):
                        x1, y1 = coordinates[i]
                        x2, y2 = coordinates[i + 1]
                        total_distance += calculate_distance(x1, y1, x2, y2)  # Calculate distance for each segment
                    
                    parameters_data[activity]["lines_length"][lane_name] = total_distance

                user_data.active_activities_for_cleaning[activity]=None
            
            else:
                zone_data={zone: Polygon(coords) for zone, coords in details["zones"].items()}
                parameters_data=details["parameters"]
                ActivityClass = load_activity_class(activity)
                if not ActivityClass:
                    continue

                # Pass user_data as parent
                activity_instance = ActivityClass(user_data,zone_data,parameters_data)
                active_instances.append(activity_instance)

                # Register available methods from the activity instance
                if isinstance(activity_instance, object):
                    # Register run() if available
                    run_method = getattr(activity_instance, "run", None)
                    if callable(run_method):
                        active_methods.append(run_method)

                    # Register cleaning() if available
                    cleaning_method = getattr(activity_instance, "cleaning", None)
                    if callable(cleaning_method):
                        user_data.active_activities_for_cleaning[activity] = cleaning_method
        else:
            print("This Activity is not Active Right Now")

    #Assigning Zones Data to Activity Instance
    user_data.zone_data=zones_data
    user_data.parameters_data = parameters_data
    user_data.violation_id_data = violation_id_data
    for activity in active_activities:
        if activity=="traffic_overspeeding_distancewise":
            # Retrieve the method from the Activities instance if it exists
            activity_func = getattr(user_data, activity, None)
            if callable(activity_func):
                active_methods.append(activity_func)
            else:
                # Activity not recognized - silently continue
                pass

    user_data.active_methods=active_methods
    
    app = GStreamerDetectionApp(app_callback, user_data)
    
    try:
        print("🚀 Starting SVDS detection system...")
        print("💡 Press Ctrl+C to stop gracefully")
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("🧹 Final cleanup...")
        cleanup_resources()
