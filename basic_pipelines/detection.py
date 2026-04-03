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
import logging

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

# yoloe handler
from basic_pipelines.yoloe_handler import YOLOEHandler

# anpr handler
from basic_pipelines.anpr_handler import ANPRHandler

# FACE RECOGNITION
from basic_pipelines.face_recognition_handler import FaceRecognitionHandler

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
        self.detection_score = None
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

        # Pose estimation variables
        self.pose_hef_path = None
        self.pose_keypoints = []          # List of keypoint lists per person
        self.pose_tracker_ids = []        # Tracker IDs for persons with pose
        self.pose_detection_boxes = []    # Bounding boxes for persons with pose
        self.pose_classes = []            # Should be all "person"
        
        # YOLOE Handler Initialization
        self.yoloe_handler = None
        
        # Centralized YOLOE control
        self.yoloe_activities = {}   
        self.yoloe_condition_labels = []  
        self.yoloe_intervals = []    
        self.yoloe_results = {}      
        self.yoloe_last_run_global = 0.0  
        self.yoloe_start_ts = None   
        self.yoloe_lock = Lock()
        self.yoloe_thread = None
        self.yoloe_running = False
        
        # ANPR Handler
        self.anpr_handler = None        # set in main() via _init_anpr_from_config
        self.anpr_activities  = {}      # {activity_name: {"interval": int, "next_run_ts": float}}
        self.anpr_results     = {}      # {activity_name: {"plate": str, "last_run": float}}
        self.anpr_lock        = Lock()  # guards anpr_results + anpr_activities
        self.anpr_running     = False   # controls the background thread
        self.anpr_thread      = None
        
        # Face Recognition Handler #
        self.facerec_handler   = None     # set in main() after config is loaded
        self.facerec_activities = {}      # {activity_name: {"interval": int, "next_run_ts": float}}
        self.facerec_results    = {}      # {tracker_id: {"person_name": str, "confidence": float, "last_run": float}}
        self.facerec_lock       = Lock()  # guards facerec_results + facerec_activities
        self.facerec_running    = False
        self.facerec_thread     = None
        
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
        if hasattr(self, 'take_video') and self.take_video and self.recorder is not None:
            try:
                video_bytes = await asyncio.wait_for(
                    self.main_loop.run_in_executor(
                        self.thread_pool,
                        self.recorder.generate_video_bytes
                    ),
                    timeout=20.0  # 10-second timeout
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
            "color": "#FF0000"
        }

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
                
# ----------------------------- YOLOE CENTRAL CONTROL ---------------------------------
# ============================================================================
#  METHOD: _init_yoloe_from_config
# ============================================================================
 
    def _init_yoloe_from_config(self, config: dict):
        """
        Reads YOLOE config for all active activities and populates
        self.yoloe_activities.
     
        No network calls at startup — the server does not need to be online here.
        The circuit breaker in yoloe_handler.py handles offline gracefully.
        """
        self.yoloe_activities       = {}
        self.yoloe_condition_labels = []
        self.yoloe_intervals        = []
        now_ts = time.time()
     
        activities_data   = config.get("activities_data",   {})
        active_activities = config.get("active_activities", [])
     
        for activity_name, details in activities_data.items():
            if activity_name not in active_activities:
                continue
     
            params = details.get("parameters", {})
            if not params.get("yoloe", 0):
                continue
     
            # ── Mode ──────────────────────────────────────────────────────────── #
            mode = str(params.get("yoloe_mode", "text")).lower().strip()
            if mode not in ("text", "visual", "both"):
                logging.warning(
                    f"[YOLOE] '{activity_name}': unknown yoloe_mode='{mode}' "
                    f"→ defaulting to 'text'"
                )
                mode = "text"
     
            # ── Text labels ────────────────────────────────────────────────────── #
            condition_labels = params.get("condition_label", [])
            if not isinstance(condition_labels, list):
                condition_labels = []
     
            if mode in ("text", "both") and not condition_labels:
                logging.warning(
                    f"[YOLOE] '{activity_name}': mode='{mode}' but no "
                    f"condition_label provided"
                )
                if mode == "text":
                    continue          # text-only with no labels = nothing to do
                mode = "visual"       # degrade both → visual
     
            # ── Interval / confidence ─────────────────────────────────────────── #
            interval   = int(params.get("yoloe_interval",   0) or 0)
            confidence = float(params.get("yoloe_confidence", 0.0) or 0.0)
            if interval <= 0:
                logging.warning(
                    f"[YOLOE] '{activity_name}': yoloe_interval={interval} ≤ 0 — skipping"
                )
                continue
     
            self.yoloe_activities[activity_name] = {
                "mode":             mode,
                "condition_labels": condition_labels,
                "activity_id":      activity_name,   # key for visual_prompts.json lookup
                "interval":         interval,
                "confidence":       confidence,
                "next_run_ts":      now_ts + interval,
            }
            self.yoloe_intervals.append(interval)
     
            if mode in ("text", "both"):
                self.yoloe_condition_labels.extend(condition_labels)
     
        # Deduplicate text labels
        self.yoloe_condition_labels = sorted(set(self.yoloe_condition_labels))
     
        if self.yoloe_intervals:
            self.yoloe_start_ts        = now_ts
            self.yoloe_last_run_global = 0.0
     
        if self.yoloe_activities:
            summary = {a: i["mode"] for a, i in self.yoloe_activities.items()}
            logging.info(f"[YOLOE] Init complete: {summary}")
            logging.info(f"[YOLOE] Text labels: {self.yoloe_condition_labels}")
        else:
            logging.info("[YOLOE] No activities configured.")
 
 
# ============================================================================
#  METHOD: _yoloe_should_run_for_activity
# ============================================================================
 
    def _yoloe_should_run_for_activity(self, activity_name: str, now_ts: float) -> bool:
        info = self.yoloe_activities.get(activity_name)
        if not info:
            return False
        interval = info.get("interval", 0)
        if interval <= 0:
            return False
        return now_ts >= info.get("next_run_ts", 0.0)
 
 
# ============================================================================
#  METHOD: _yoloe_thread_loop
# ============================================================================
 
    def _yoloe_thread_loop(self):
        """
        Daemon thread: checks each activity's interval, grabs the latest Hailo
        frame, routes to text / visual / both via yoloe_handler, stores result.
     
        Mode routing:
          text   → yoloe_handler.text_prompt(frame, labels)
          visual → yoloe_handler.visual_prompt(frame, activity_id)
          both   → yoloe_handler.both_prompt(frame, labels, activity_id)
        """
        if not self.yoloe_activities:
            return
     
        base_sleep = 1.0
     
        while self.yoloe_running:
            now_ts = time.time()
            try:
                # Snapshot the latest Hailo frame
                image = None
                with self.yoloe_lock:
                    image = self.model_image.copy() if self.model_image is not None else None
     
                if image is None:
                    time.sleep(base_sleep)
                    continue
     
                for act_name, info in list(self.yoloe_activities.items()):
                    if not self._yoloe_should_run_for_activity(act_name, now_ts):
                        continue
     
                    mode   = info["mode"]
                    conf   = info.get("confidence", 0.05)
                    act_id = info["activity_id"]
                    labels = info.get("condition_labels", [])
     
                    # ── Route to the correct handler method ────────────────────── #
                    api_result = None
                    try:
                        if mode == "text":
                            api_result = self.yoloe_handler.text_prompt(
                                image, labels, conf=conf
                            )
                        elif mode == "visual":
                            api_result = self.yoloe_handler.visual_prompt(
                                image, act_id, conf=conf
                            )
                        elif mode == "both":
                            api_result = self.yoloe_handler.both_prompt(
                                image, labels, act_id, conf=conf
                            )
                    except Exception as e:
                        logging.error(f"[YOLOE] '{act_name}' {mode} call failed: {e}")
     
                    # ── Store per-activity result ──────────────────────────────── #
                    if api_result is not None:
                        with self.yoloe_lock:
                            self.yoloe_results[act_name] = {
                                "result":   api_result,
                                "last_run": now_ts,
                            }
                            self.yoloe_last_run_global = now_ts
     
                    # ── Advance next_run_ts ────────────────────────────────────── #
                    with self.yoloe_lock:
                        interval = info.get("interval", 0)
                        if interval > 0:
                            next_ts = info.get("next_run_ts", now_ts)
                            while next_ts <= now_ts:
                                next_ts += interval
                            info["next_run_ts"] = next_ts
     
            except Exception as e:
                logging.error(f"[YOLOE] Thread loop error: {e}")
     
            time.sleep(base_sleep)
 
 
# ============================================================================
#  METHOD: get_yoloe_result
# ============================================================================
 
    def get_yoloe_result(self, activity_name: str) -> dict | None:
        """
        Thread-safe accessor for activity modules.
     
        Returns:
            {
              "result":   <server JSON with detections, image_base64, mode, ...>
              "last_run": <unix timestamp>
            }
            or None if no result yet for this activity.
     
        Usage in an activity module:
            data = user_data.get_yoloe_result("ppe_detection")
            if data:
                for det in data["result"].get("detections", []):
                    label  = det["prompt"]     # e.g. "hard_hat"
                    conf   = det["confidence"]
                    bbox   = det["bounding_box"]
                    source = det.get("source", "")  # "text" / "visual" / ""
        """
        with self.yoloe_lock:
            return self.yoloe_results.get(activity_name)
        

# ============================================================================
#  ANPR CENTRAL CONTROL
# ============================================================================
 
    def _init_anpr_from_config(self, config: dict):
        """
        Reads ANPR config for each active activity that has  "anpr": 1.
 
        Expected JSON shape inside an activity's  parameters  block:
            {
                "anpr": 1,
                "anpr_interval": 30      ← seconds between full-frame ANPR sweeps
                                           (0 or absent  → event-driven only, no thread)
            }
 
        Activities with  anpr_interval > 0  are added to self.anpr_activities
        so the background thread runs them on a timer (Mode A — interval).
 
        Activities with  anpr_interval == 0  still get access to
        self.anpr_handler for on-demand / event-driven calls (Mode B).
 
        No network calls are made here.  The circuit breaker in
        anpr_handler.py handles a dead server gracefully at call time.
        """
        self.anpr_activities = {}
        now_ts = time.time()
 
        activities_data   = config.get("activities_data",   {})
        active_activities = config.get("active_activities", [])
 
        for activity_name, details in activities_data.items():
            if activity_name not in active_activities:
                continue
 
            params = details.get("parameters", {})
 
            # Only configure activities that explicitly opt in to ANPR
            if not params.get("anpr", 0):
                continue
 
            interval = int(params.get("anpr_interval", 0) or 0)
 
            if interval > 0:
                # Mode A: interval-based background sweeps on the full frame
                self.anpr_activities[activity_name] = {
                    "interval":    interval,
                    "next_run_ts": now_ts + interval,
                }
                logging.info(
                    f"[ANPR] '{activity_name}' registered for interval-based "
                    f"sweeps every {interval}s."
                )
            else:
                # Mode B: event-driven only — no background sweep needed,
                # the activity calls self.parent.anpr_handler.process_frame()
                # from its own one-off thread.
                logging.info(
                    f"[ANPR] '{activity_name}' registered for event-driven "
                    f"(on-demand) ANPR only."
                )
 
        if self.anpr_activities:
            logging.info(
                f"[ANPR] Interval activities: "
                f"{list(self.anpr_activities.keys())}"
            )
        else:
            logging.info("[ANPR] No interval-based ANPR activities configured.")
 
 
    def _anpr_thread_loop(self):
        """
        Daemon thread: fires ANPR on the latest Hailo full frame at each
        activity's configured interval and stores the result in self.anpr_results.
 
        This is Mode A (interval-based).  Activities read results via:
            plate = self.parent.anpr_results.get("MyActivity", {}).get("plate")
 
        Mode B (event-driven) never goes through this thread — each violation
        spawns its OWN one-off thread that calls anpr_handler.process_frame()
        directly (see AnprTest in activities.py).
        """
        if not self.anpr_activities:
            return
 
        base_sleep = 1.0   # poll interval in seconds
 
        while self.anpr_running:
            now_ts = time.time()
            try:
                # ── Snapshot the latest Hailo frame (thread-safe) ──────────── #
                frame = None
                with self.anpr_lock:
                    if self.model_image is not None:
                        frame = self.model_image.copy()
 
                if frame is None:
                    time.sleep(base_sleep)
                    continue
 
                # ── Check each interval-configured activity ────────────────── #
                for act_name, info in list(self.anpr_activities.items()):
                    if now_ts < info.get("next_run_ts", 0.0):
                        continue   # not time yet
 
                    plate = None
                    try:
                        plate = self.anpr_handler.process_frame(
                            frame,
                            activity_name=act_name,
                            extra_meta={"mode": "interval"},
                        )
                    except Exception as exc:
                        logging.error(
                            f"[ANPR] Interval call for '{act_name}' failed: {exc}"
                        )
 
                    # ── Store result ───────────────────────────────────────── #
                    if plate is not None:
                        with self.anpr_lock:
                            self.anpr_results[act_name] = {
                                "plate":    plate,
                                "last_run": now_ts,
                            }
 
                    # ── Advance next_run_ts (skip missed windows) ──────────── #
                    with self.anpr_lock:
                        interval = info.get("interval", 0)
                        if interval > 0:
                            next_ts = info.get("next_run_ts", now_ts)
                            while next_ts <= now_ts:
                                next_ts += interval
                            info["next_run_ts"] = next_ts
 
            except Exception as exc:
                logging.error(f"[ANPR] Thread loop error: {exc}")
 
            time.sleep(base_sleep)
            
            
# ============================================================================
#  FACE RECOGNITION CENTRAL CONTROL
# ============================================================================
 
    def _init_facerec_from_config(self, config: dict):
        """
        Reads face-rec config for each active activity that has "face_rec": 1.
 
        Expected JSON inside the activity's parameters block:
        {
            "face_rec": 1,
            "facerec_interval": 5    ← seconds between batch sweeps (Mode A)
                                       0 or absent = event-driven only (Mode B)
        }
 
        Mode A (interval > 0): background thread grabs ALL visible person
        crops every N seconds and batch-sends them to the API.
 
        Mode B (interval == 0): the activity itself decides when to fire —
        e.g. only on first appearance of a new tracker_id.
        """
        self.facerec_activities = {}
        now_ts = time.time()
 
        activities_data   = config.get("activities_data",   {})
        active_activities = config.get("active_activities", [])
 
        for activity_name, details in activities_data.items():
            if activity_name not in active_activities:
                continue
 
            params = details.get("parameters", {})
            if not params.get("face_rec", 0):
                continue
 
            interval = int(params.get("facerec_interval", 0) or 0)
 
            if interval > 0:
                self.facerec_activities[activity_name] = {
                    "interval":    interval,
                    "next_run_ts": now_ts + interval,
                }
                logging.info(
                    f"[FaceRec] '{activity_name}' interval sweep every {interval}s."
                )
            else:
                logging.info(
                    f"[FaceRec] '{activity_name}' event-driven mode only."
                )
 
        if self.facerec_activities:
            logging.info(
                f"[FaceRec] Interval activities: "
                f"{list(self.facerec_activities.keys())}"
            )
        else:
            logging.info("[FaceRec] No interval activities configured.")
 
 
    def _facerec_thread_loop(self):
        """
        Daemon thread — Mode A (interval-based).
 
        Every N seconds:
          1. Snapshot the latest Hailo frame + detection data.
          2. Find all "person" detections.
          3. Crop the face region (top 35% of each person bbox).
          4. Skip crops that are too small or where the person is facing away.
          5. Batch-send all valid crops in ONE API call.
          6. Store per-tracker results in self.facerec_results.
 
        Activities read results via:
            result = self.parent.facerec_results.get(tracker_id)
            # {"person_name": "Alice", "confidence": 97.8, "last_run": float}
        """
        if not self.facerec_activities:
            return
 
        MIN_FACE_PX = 40   # ignore face crops smaller than 40×40 — not useful
        base_sleep  = 0.5
 
        while self.facerec_running:
            now_ts = time.time()
            try:
                # ── Snapshot frame data under lock ────────────────────────── #
                frame = classes = tracker_ids = boxes = None
                ratio = padx = pady = orig_h = orig_w = None
 
                with self.facerec_lock:
                    if self.model_image is not None:
                        frame       = self.model_image.copy()
                        classes     = list(self.classes or [])
                        tracker_ids = list(self.tracker_ids or [])
                        boxes       = list(self.detection_boxes or [])
                        ratio       = self.ratio  or 1.0
                        padx        = self.padx   or 0
                        pady        = self.pady   or 0
                        orig_h      = self.original_height or frame.shape[0]
                        orig_w      = self.original_width  or frame.shape[1]
 
                if frame is None:
                    time.sleep(base_sleep)
                    continue
 
                for act_name, info in list(self.facerec_activities.items()):
                    if now_ts < info.get("next_run_ts", 0.0):
                        continue
 
                    # ── Collect person face crops ──────────────────────────── #
                    face_crops = []
                    for i, cls in enumerate(classes):
                        if cls != "person":
                            continue
                        tid = tracker_ids[i]
                        box = boxes[i]
 
                        # Reverse letterbox → original pixel coords
                        x1 = max(0, int((box[0] - padx) / ratio))
                        y1 = max(0, int((box[1] - pady) / ratio))
                        x2 = min(orig_w, int((box[2] - padx) / ratio))
                        y2 = min(orig_h, int((box[3] - pady) / ratio))
 
                        # Face = top 35% of person bounding box
                        face_y2 = y1 + max(1, int((y2 - y1) * 0.35))
                        face_crop = frame[y1:face_y2, x1:x2]
 
                        fh, fw = face_crop.shape[:2]
                        if fh < MIN_FACE_PX or fw < MIN_FACE_PX:
                            continue   # person too far away
 
                        face_crops.append({
                            "tracker_id": tid,
                            "crop":       face_crop.copy(),
                            "extra_meta": {"activity": act_name, "mode": "interval"},
                        })
 
                    if not face_crops:
                        continue
 
                    # ── Batch API call ─────────────────────────────────────── #
                    try:
                        results = self.facerec_handler.match_faces(
                            face_crops,
                            activity_name=act_name,
                        )
                    except Exception as exc:
                        logging.error(
                            f"[FaceRec] Interval call '{act_name}' failed: {exc}"
                        )
                        results = []
 
                    # ── Store per-tracker results ──────────────────────────── #
                    with self.facerec_lock:
                        for r in results:
                            if r.get("status") == "match_found":
                                self.facerec_results[r["tracker_id"]] = {
                                    "person_name": r["person_name"],
                                    "confidence":  r["confidence"],
                                    "last_run":    now_ts,
                                }
 
                    # ── Advance next_run_ts ────────────────────────────────── #
                    with self.facerec_lock:
                        interval = info.get("interval", 0)
                        if interval > 0:
                            next_ts = info.get("next_run_ts", now_ts)
                            while next_ts <= now_ts:
                                next_ts += interval
                            info["next_run_ts"] = next_ts
 
            except Exception as exc:
                logging.error(f"[FaceRec] Thread loop error: {exc}")
 
            time.sleep(base_sleep)
 
 
    def get_facerec_result(self, tracker_id: int) -> dict | None:
        """
        Thread-safe accessor for activity modules.
 
        Returns:
            {"person_name": str, "confidence": float, "last_run": float}
            or None if tracker_id not yet identified.
        """
        with self.facerec_lock:
            return self.facerec_results.get(tracker_id)

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
    # Check if the buffer is valid and of correct type
    if buffer is None or not isinstance(buffer, Gst.Buffer):
        return Gst.PadProbeReturn.OK

    # Simple frame counter for external monitoring
    user_data.frame_monitor_count += 1
    
    # Reset counter if it reaches 1000 to prevent overflow
    if user_data.frame_monitor_count >= 1000:
        user_data.frame_monitor_count = 0
        #print("INFO: Frame monitor counter reset to prevent overflow")

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    # print("frame type: ",frame_type)
    
    # ============================================
    # CASE 1: ORIGINAL FRAME CAPTURE
    # ============================================
    if frame_type == "original":
        if format is not None and width is not None and height is not None:
            # Get video frame
            user_data.image = get_numpy_from_buffer(buffer, format, width, height)
            if user_data.original_height is None:
                user_data.original_height = height
                user_data.original_width = width
    # ============================================
    # CASE 2: POSE ESTIMATION RESULTS
    # ============================================
    elif frame_type == "pose_estimated": # try make this minimal since the same work is done by processing as well
        if format is None or width is None or height is None:
            return Gst.PadProbeReturn.OK
            
        # Get detections from buffer (pose model outputs person detections + keypoints)
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        if len(detections) == 0:
            return Gst.PadProbeReturn.OK
        
        # Calculate scaling ratios
        ratio = min(width / user_data.original_width, height / user_data.original_height)
        padx = int((width - user_data.original_width * ratio) / 2)
        pady = int((height - user_data.original_height * ratio) / 2)
        
         # Get keypoint name-to-index mapping
        keypoints = get_keypoints()
        
        # Temporary storage for this frame's pose data
        pose_xyxys = []
        pose_keypoints = []
        pose_class_names = []
        
        for detection in detections:
            label = detection.get_label()
            bbox = detection.get_bbox()
            
            # Scale bounding box to original coordinates
            x_min = max(int((bbox.xmin() * width - padx) / ratio), 0)
            y_min = max(int((bbox.ymin() * height - pady) / ratio), 0)
            x_max = min(int((bbox.xmax() * width - padx) / ratio), user_data.original_width)
            y_max = min(int((bbox.ymax() * height - pady) / ratio), user_data.original_height)
            
            # Get pose landmarks
            pose_landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            
            if len(pose_landmarks) > 0:
                points = pose_landmarks[0].get_points()
                scaled_keypoints = []
                
                # Scale each keypoint to original coordinates
                for pose_id in keypoints.keys():
                    keypoint_index = keypoints[pose_id]
                    if keypoint_index < len(points):
                        kp = points[keypoint_index]
                        kp_x = min(max(int((kp.x() * width - padx) / ratio), 0), user_data.original_width)
                        kp_y = min(max(int((kp.y() * height - pady) / ratio), 0), user_data.original_height)
                        conf = kp.confidence()
                        scaled_keypoints.append((kp_x, kp_y, conf))
                
                # Store this person's pose data
                pose_xyxys.append([x_min, y_min, x_max, y_max])
                pose_keypoints.append(scaled_keypoints)
                pose_class_names.append(label)
        
        # Update user_data with latest pose results
        user_data.pose_detection_boxes = pose_xyxys
        user_data.pose_keypoints = pose_keypoints
        user_data.pose_classes = pose_class_names
        
        # Optional: Store in pose_results dict by tracker_id if you have tracker for pose
        # For now, pose branch doesn't have tracker, so we just store the latest frame's data
    
    # ============================================
    # CASE 3: DETECTION RESULTS (Main Pipeline)
    # ============================================
    elif frame_type=="processing":
        # Your existing detection processing code
        frame = None
        if format is not None and width is not None and height is not None:
            try:
                if not isinstance(buffer, Gst.Buffer):
                    return Gst.PadProbeReturn.OK

                buf_timestamp = buffer.pts
                ts = buf_timestamp / Gst.SECOND
                user_data.time_stamp.append(ts)

                frame = get_numpy_from_buffer(buffer, format, width, height)
                if hasattr(user_data, 'pose_keypoints') and user_data.pose_keypoints:
                    # Draw the lines directly onto the frame
                    frame = draw_skeleton_on_frame(frame, user_data.pose_keypoints)
                    
                user_data.model_image = frame
                if user_data.recorder is not None:
                    user_data.recorder.add_frame(frame)
            except TypeError:
                return Gst.PadProbeReturn.OK
        
        # Get the detections from the buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # Parse the detections
        detection_count = 0
        xyxys = []
        class_ids = []
        class_names = []
        confidences = []
        tracker_ids = []
        anchor_points_original = []
        
        if user_data.ratio is None and user_data.original_width is not None:
            user_data.ratio = min(width / user_data.original_width, height / user_data.original_height)
            user_data.padx = int((width - user_data.original_width * user_data.ratio) / 2)
            user_data.pady = int((height - user_data.original_height * user_data.ratio) / 2)
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
                    
                    x_min = max(int((bbox.xmin() * width - user_data.padx) / user_data.ratio), 0)
                    y_min = max(int((bbox.ymin() * height - user_data.pady) / user_data.ratio), 0)
                    x_max = max(int((bbox.xmax() * width - user_data.padx) / user_data.ratio), 0)
                    y_max = max(int((bbox.ymax() * height - user_data.pady) / user_data.ratio), 0)

                    # Appending the results from Detections
                    xyxys.append([x_min, y_min, x_max, y_max])
                    class_ids.append(detection.get_class_id())
                    class_names.append(label)
                    confidences.append(confidence)
                    anchor_points_original.append((((x_min + x_max) / 2), (y_max)))
                
                detection_count += 1
            
            # Updating the Activities Instance
            user_data.detection_boxes = xyxys
            user_data.classes = class_names
            user_data.tracker_ids = tracker_ids
            user_data.anchor_points_original = anchor_points_original
            user_data.detection_score = confidences
            
            # Last n frame tracker ids
            user_data.LNFCTI.append(tracker_ids)
            user_data.last_n_frame_tracker_ids = get_unique_tracker_ids(user_data.LNFCTI)

            # Clean every 120 seconds
            if int((time.time() - user_data.cleaning_time_for_events)) > 120:
                user_data.cleaning_events_data_with_last_frames()
                user_data.cleaning_time_for_events = time.time()

            # Run activity methods
            for method in user_data.active_methods:
                method()

    return Gst.PadProbeReturn.OK

def draw_skeleton_on_frame(image, pose_keypoints, confidence_threshold=0.4):
    """
    Draws skeleton using the specific keys from get_keypoints()
    """
    if image is None or not pose_keypoints:
        return image

    # Define connections (bone lines) based on your get_keypoints() keys
    CONNECTIONS = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
    ]

    # Get the mapping from your existing function
    keypoint_map = get_keypoints()
    
    # Colors (BGR)
    COLOR_POINT = (0, 255, 255) # Yellow
    COLOR_LINE = (0, 255, 0)    # Green

    for person in pose_keypoints:
        # person is a list of (x, y, conf)
        
        # 1. Draw Lines (Bones)
        for start_name, end_name in CONNECTIONS:
            idx_start = keypoint_map.get(start_name)
            idx_end = keypoint_map.get(end_name)
            
            # Ensure indices exist in the detection
            if idx_start is not None and idx_end is not None and \
               idx_start < len(person) and idx_end < len(person):
                
                x1, y1, c1 = person[idx_start]
                x2, y2, c2 = person[idx_end]
                
                if c1 > confidence_threshold and c2 > confidence_threshold:
                    try:
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_LINE, 2)
                    except Exception:
                        pass

        # 2. Draw Points (Joints)
        for i, (x, y, conf) in enumerate(person):
            if conf > confidence_threshold:
                try:
                    cv2.circle(image, (int(x), int(y)), 4, COLOR_POINT, -1)
                except Exception:
                    pass

    return image
    
def get_keypoints():
    return {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }
    
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

# ~ def load_activity_class(activity_name):
    # ~ """Load activity class dynamically from basic_pipelines/activities/"""
    # ~ try:
        # ~ module = importlib.import_module(f"basic_pipelines.activities.{activity_name}")
        # ~ return getattr(module, activity_name)  # Class name must match filename
    # ~ except Exception as e:
        # ~ print(f"❌ Failed to load {activity_name}: {e}")
        # ~ return None
        
    
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
        kafka_vars["broker_failover_timeout"] = int(os.getenv("BROKER_FAILOVER_TIMEOUT", 30))
        kafka_vars["send_analytics_pipeline"] = os.getenv("SEND_ANALYTICS_PIPELINE", "variphianalytics")
        kafka_vars["send_events_pipeline"] = os.getenv("SEND_EVENTS_PIPELINE", "variphievents")
        kafka_vars["log_topic"] = os.getenv("LOG_TOPIC", "error_log")

    # AWS S3 overrides (primary / secondary)
    s3 = kafka_vars.get("AWS_S3", {})
    # Helpers to read env with backward-compatible aliases
    def _env(*names, default=None):
        for n in names:
            v = os.getenv(n)
            if v is not None and v != "":
                return v
        return default

    if _env("AWS_PRIMARY_KEY", "AWS_PRIMARY_KEY_ID") and _env("AWS_PRIMARY_SECRET"):
        s3.setdefault("primary", {})
        s3["primary"]["aws_access_key_id"] = _env("AWS_PRIMARY_KEY", "AWS_PRIMARY_KEY_ID")
        s3["primary"]["aws_secret_access_key"] = _env("AWS_PRIMARY_SECRET")
        s3["primary"]["end_point_url"] = _env("AWS_PRIMARY_ENDPOINT")
        s3["primary"]["region_name"] = _env("AWS_PRIMARY_REGION", default="ap-south-1")
        s3["primary"]["BUCKET_NAME"] = _env("AWS_PRIMARY_BUCKET_NAME", "AWS_PRIMARY_BUCKET", default="arrestovideos")
        s3["primary"]["org_img_fn"] = _env("AWS_PRIMARY_ORG_IMG_FN", "AWS_PRIMARY_ORG_IMG_PATH", default="violationoriginalimages/")
        s3["primary"]["video_fn"] = _env("AWS_PRIMARY_VIDEO_FN", "AWS_PRIMARY_VIDEO_PATH", default="videoclips/")
        s3["primary"]["cgi_fn"] = _env("AWS_PRIMARY_CGI_FN", "AWS_PRIMARY_CGI_PATH", default="cgisnapshots/")
   
    if _env("AWS_SECONDARY_KEY", "AWS_SECONDARY_KEY_ID") and _env("AWS_SECONDARY_SECRET"):
        s3.setdefault("secondary", {})
        s3["secondary"]["aws_access_key_id"] = _env("AWS_SECONDARY_KEY", "AWS_SECONDARY_KEY_ID")
        s3["secondary"]["aws_secret_access_key"] = _env("AWS_SECONDARY_SECRET")
        s3["secondary"]["end_point_url"] = _env("AWS_SECONDARY_ENDPOINT")
        s3["secondary"]["region_name"] = _env("AWS_SECONDARY_REGION", default="ap-south-1")
        s3["secondary"]["BUCKET_NAME"] = _env("AWS_SECONDARY_BUCKET_NAME", "AWS_SECONDARY_BUCKET", default="arrestovideos")
        s3["secondary"]["org_img_fn"] = _env("AWS_SECONDARY_ORG_IMG_FN", "AWS_SECONDARY_ORG_IMG_PATH", default="violationoriginalimages/")
        s3["secondary"]["video_fn"] = _env("AWS_SECONDARY_VIDEO_FN", "AWS_SECONDARY_VIDEO_PATH", default="videoclips/")
        s3["secondary"]["cgi_fn"] = _env("AWS_SECONDARY_CGI_FN", "AWS_SECONDARY_CGI_PATH", default="cgisnapshots/")

    if _env("API_AWS_PRIMARY_KEY", "API_AWS_PRIMARY_KEY_ID") and _env("API_AWS_PRIMARY_SECRET"):
        s3.setdefault("api_primary", {})
        s3["api_primary"]["aws_access_key_id"] = _env("API_AWS_PRIMARY_KEY", "API_AWS_PRIMARY_KEY_ID")
        s3["api_primary"]["aws_secret_access_key"] = _env("API_AWS_PRIMARY_SECRET")
        s3["api_primary"]["end_point_url"] = _env("API_AWS_PRIMARY_ENDPOINT")
        s3["api_primary"]["region_name"] = _env("API_AWS_PRIMARY_REGION", default="ap-south-1")
        s3["api_primary"]["BUCKET_NAME"] = _env("API_AWS_PRIMARY_BUCKET_NAME", "API_AWS_PRIMARY_BUCKET", default="arrestovideos")
        s3["api_primary"]["org_img_fn"] = _env("API_AWS_PRIMARY_ORG_IMG_FN", "API_AWS_PRIMARY_ORG_IMG_PATH", default="violationoriginalimages/")
        s3["api_primary"]["video_fn"] = _env("API_AWS_PRIMARY_VIDEO_FN", "API_AWS_PRIMARY_VIDEO_PATH", default="videoclips/")
        s3["api_primary"]["cgi_fn"] = _env("API_AWS_PRIMARY_CGI_FN", "API_AWS_PRIMARY_CGI_PATH", default="cgisnapshots/")

    if _env("API_AWS_SECONDARY_KEY", "API_AWS_SECONDARY_KEY_ID") and _env("API_AWS_SECONDARY_SECRET"):
        s3.setdefault("api_secondary", {})
        s3["api_secondary"]["aws_access_key_id"] = _env("API_AWS_SECONDARY_KEY", "API_AWS_SECONDARY_KEY_ID")
        s3["api_secondary"]["aws_secret_access_key"] = _env("API_AWS_SECONDARY_SECRET")
        s3["api_secondary"]["end_point_url"] = _env("API_AWS_SECONDARY_ENDPOINT")
        s3["api_secondary"]["region_name"] = _env("API_AWS_SECONDARY_REGION", default="ap-south-1")
        s3["api_secondary"]["BUCKET_NAME"] = _env("API_AWS_SECONDARY_BUCKET_NAME", "API_AWS_SECONDARY_BUCKET", default="arrestovideos")
        s3["api_secondary"]["org_img_fn"] = _env("API_AWS_SECONDARY_ORG_IMG_FN", "API_AWS_SECONDARY_ORG_IMG_PATH", default="violationoriginalimages/")
        s3["api_secondary"]["video_fn"] = _env("API_AWS_SECONDARY_VIDEO_FN", "API_AWS_SECONDARY_VIDEO_PATH", default="videoclips/")
        s3["api_secondary"]["cgi_fn"] = _env("API_AWS_SECONDARY_CGI_FN", "API_AWS_SECONDARY_CGI_PATH", default="cgisnapshots/")
    
    s3["s3_failover_timeout"] = int(os.getenv("S3_FAILOVER_TIMEOUT", 30))
    s3["upload_retries"] = int(os.getenv("S3_UPLOAD_RETRIES", 3))

    kafka_vars["AWS_S3"] = s3
    config["kafka_variables"] = kafka_vars

    # Setup the Hef-path with path existence check
    try:
        hef_path = config.get("default_arguments", {}).get("hef_path")
        labels_json = config.get("default_arguments", {}).get("labels-json")
        pose_hef_path = config.get("default_arguments", {}).get("pose_hef_path")
        
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
            
        # Check if pose_hef path exists and is not None/empty
        if pose_hef_path and pose_hef_path != "None" and os.path.exists(pose_hef_path):
            user_data.pose_hef_path = pose_hef_path
            # print(f"✅ Pose HEF configured: {pose_hef_path}")
        else:
            user_data.pose_hef_path = None
            print(f"ℹ️ Pose HEF not configured or missing: {pose_hef_path}")
            
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
   
    # ==========================================================
    # Initialize centralized YOLOE control
    # ==========================================================
    try:
        user_data.yoloe_handler = YOLOEHandler(config)
        # Link the results queue if you want YOLOE results sent to Kafka
        # user_data.yoloe_handler.set_results_queue(results_events_queue) 
        
        user_data._init_yoloe_from_config(config)
        if user_data.yoloe_activities:
            user_data.yoloe_running = True
            user_data.yoloe_thread = Thread(target=user_data._yoloe_thread_loop, daemon=True)
            user_data.yoloe_thread.start()
            modes = {a: i["mode"] for a, i in user_data.yoloe_activities.items()}
            print(f"✅ YOLOE scheduler started: {modes}")
        else:
            print("ℹ️ YOLOE: no activities configured.")
    except Exception as e:
        print(f"❌ Failed to initialize YOLOE: {e}")

    # ==========================================================
    # Initialize centralized ANPR control
    # ==========================================================
    try:
        user_data.anpr_handler = ANPRHandler(config)
        user_data._init_anpr_from_config(config)
 
        if user_data.anpr_activities:
            # Mode A activities exist → start the background sweep thread
            user_data.anpr_running = True
            user_data.anpr_thread  = Thread(
                target=user_data._anpr_thread_loop, daemon=True
            )
            user_data.anpr_thread.start()
            print(
                f"✅ ANPR interval scheduler started: "
                f"{list(user_data.anpr_activities.keys())}"
            )
        else:
            # Only Mode B (event-driven) activities or none at all
            print("ℹ️ ANPR: handler ready (event-driven mode only — no interval thread).")
    except Exception as exc:
        print(f"❌ Failed to initialize ANPR: {exc}")
        
    # ==========================================================
    # Initialize centralized Face Recognition control
    # ==========================================================
    try:
        user_data.facerec_handler = FaceRecognitionHandler(config)
        user_data._init_facerec_from_config(config)
 
        if user_data.facerec_activities:
            user_data.facerec_running = True
            user_data.facerec_thread  = Thread(
                target=user_data._facerec_thread_loop, daemon=True
            )
            user_data.facerec_thread.start()
            print(
                f"✅ FaceRec interval scheduler started: "
                f"{list(user_data.facerec_activities.keys())}"
            )
        else:
            print("ℹ️ FaceRec: handler ready (event-driven mode only).")
    except Exception as exc:
        print(f"❌ Failed to initialize FaceRecognition: {exc}")
        
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
    results_analytics_queue = queue.Queue(maxsize=100)
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
    # ~ zones_data = {}
    # ~ parameters_data={}
    # ~ violation_id_data={}

    # ~ active_instances=[]
    # ~ active_methods=[]
    # ~ for activity, details in config["activities_data"].items():
        # ~ if activity in available_activities:
            # ~ if activity in active_activities:
                # ~ zone_data={zone: Polygon(coords) for zone, coords in details["zones"].items()}
                # ~ parameters_data[activity] =details["parameters"]
                
                # ~ ActivityClass = load_activity_class(activity)
                # ~ if not ActivityClass:
                    # ~ print(f"Failed to load activity class: {activity}")
                    # ~ continue

                # ~ # Pass user_data as parent
                # ~ activity_instance = ActivityClass(user_data,zone_data,parameters_data[activity])
                # ~ active_instances.append(activity_instance)

                # ~ # Register available methods from the activity instance
                # ~ if isinstance(activity_instance, object):
                    # ~ # Register run() if available
                    # ~ run_method = getattr(activity_instance, "run", None)
                    # ~ if callable(run_method):
                        # ~ active_methods.append(run_method)

                    # ~ # Register cleaning() if available
                    # ~ cleaning_method = getattr(activity_instance, "cleaning", None)
                    # ~ if callable(cleaning_method):
                        # ~ user_data.active_activities_for_cleaning[activity] = cleaning_method
            # ~ else:
                # ~ print("This Activity is not active right now")
        # ~ else:
            # ~ print("This Activity is not available right now.")

    # ~ #Assigning Zones Data to Activity Instance
    # ~ user_data.zone_data=zones_data
    # ~ user_data.parameters_data = parameters_data
    # ~ user_data.violation_id_data = violation_id_data
    # ~ for activity in active_activities:
        # ~ if activity=="traffic_overspeeding_distancewise":
            # ~ # Retrieve the method from the Activities instance if it exists
            # ~ activity_func = getattr(user_data, activity, None)
            # ~ if callable(activity_func):
                # ~ active_methods.append(activity_func)
            # ~ else:
                # ~ # Activity not recognized - silently continue
                # ~ pass

    # ~ user_data.active_methods=active_methods
    
    # ==========================================================
    # INITIALIZE ANALYTICS ENGINE
    # ==========================================================
    active_methods = []
    user_data.active_activities_for_cleaning = {}

    try:
        # Import your new unified engine (Adjust path if it is inside basic_pipelines)
        from activities import AnalyticsEngine 
        
        print("🚀 Initializing Unified Analytics Engine...")
        user_data.analytics_engine = AnalyticsEngine(user_data, config)
        
        # Hook the engine's master 'run' and 'clean' methods into the pipeline
        active_methods.append(user_data.analytics_engine.run_all)
        user_data.active_activities_for_cleaning["all_analytics"] = user_data.analytics_engine.clean_all
        
        print(f"✅ Active Analytics Modules mapped successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Analytics Engine: {e}")

    # ~ # ==========================================================
    # ~ # HANDLE NATIVE METHODS (Traffic Overspeeding)
    # ~ # ==========================================================
    # ~ # Traffic overspeeding lives directly inside user_app_callback_class, not the engine
    # ~ for activity in active_activities:
        # ~ if activity == "traffic_overspeeding_distancewise":
            # ~ activity_func = getattr(user_data, activity, None)
            # ~ if callable(activity_func):
                # ~ active_methods.append(activity_func)
                # ~ print("✅ Registered native traffic_overspeeding_distancewise method.")

    # Assign the final list of methods to run every frame
    user_data.active_methods = active_methods
    
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
