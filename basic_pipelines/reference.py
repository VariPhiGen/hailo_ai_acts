import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
    stop_thread
)
from detection_pipeline import GStreamerDetectionApp
import boto3
from botocore.client import Config 
import uuid
import math

from shapely.geometry import Point, Polygon
import time
from datetime import datetime
import pytz
import json
import base64
import asyncio
import queue
from kafka import KafkaProducer
from threading import Thread
import supervision as sv
from supervision.detection.core import Detections as trackerinput
from collections import deque
from logging.handlers import RotatingFileHandler
import logging
from kafka.errors import KafkaError, NoBrokersAvailable
import socket
import socketio
from collections import defaultdict, deque
from scipy.spatial import distance
from scipy.stats import entropy
from itertools import combinations
import imagehash
from PIL import Image
from form_util import fill_form



# --------------------------------------------------------------------------------------------
kafka_queue=queue.Queue(maxsize=100)
#Initializing results queue
results_analytics_queue = queue.Queue(maxsize=100)
results_events_queue = queue.Queue(maxsize=100)
# Initialize a deque to store tracker_ids for the last n frames
last_n_frames_tracker_ids = deque(maxlen=30)

# AWS S3 credentials
config_file_temp=json.load(open('configuration.json'))
config_file_temp=config_file_temp["kafka_variables"]["AWS_S3"]
UPLOAD_S3_REGION = config_file_temp["region_name"]
UPLOAD_ACCESS_KEY_ID =config_file_temp["aws_access_key_id"]
UPLOAD_ACCESS_KEY_SECRET = config_file_temp["aws_secret_access_key"]
UPLOAD_S3_BUCKET_NAME = config_file_temp["BUCKET_NAME"]

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=UPLOAD_ACCESS_KEY_ID,
    aws_secret_access_key=UPLOAD_ACCESS_KEY_SECRET,
    region_name=UPLOAD_S3_REGION
)
minio_client=boto3.client(
                    "s3",
                    aws_access_key_id="root",
                    endpoint_url="http://localhost:9000",
                    config=Config(signature_version="s3v4"),
                    aws_secret_access_key="root@2025",
                    region_name="ap-south-1"
                )
# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.sensor_id=None
        self.zone_data=None
        self.violation_id_data=None
        self.parameters_data=None
        self.model_image=None
        self.image=None
        self.original_height=720
        self.original_width=1280
        self.detection_boxes=None
        self.classes=None
        self.tracker_ids=None
        self.keypoints_index =self.get_keypoints()
        self.pose_detection_boxes=None
        self.pose_classes=None
        self.pose_keypoints=None
        self.pose_tracker_ids=None
        self.prev_pos={}
        self.anchor_points=None
        self.ist_timezone=pytz.timezone('Asia/Kolkata')
        self.last_n_frame_tracker_ids=None
        self.time_stamp=deque(maxlen=10)
        self.view_transformer=None
        self.cleaning_time_for_events=time.time()
        self.prev_distances = {}

        # Workforce_efficiency
        self.workforce_efficiency_data={}
        self.desk_occupancy_data={}
        self.resource_utilization_data={}
        self.entry_exit_WLE_logs_data={}
        self.workplace_area_occupancy_data={}
        self.people_gathering_data={}
        self.ppe_data={}
        self.perimeter_monitoring_data={}
        self.climbing_data={}
        self.vehicle_interaction_data={}
        self.person_violations_data={}
        self.fire_and_smoke_data={}
        self.traffic_overspeeding_data=None
        self.anprscan_line_area=None
        self.distancewise_tracking=None
        self.traffic_overspeeding_distancewise_data=None
        self.traffic_overspeeding_speed_data=None
        self.unattended_area_data = {}
        self.camera_tampering_data = {}
        # Initialize background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=False
        )
        self.tampering_check=0
        self.prev_brightness=None
        self.ref_hash=None



        # Queue for sending Data to Kafka
        self.results_analytics_queue=None
        self.results_events_queue=None
        #Active methods
        self.active_methods=None
        self.active_activities_for_cleaning=[]

        #For Live AI Pipeline
        self.asset_config = json.load(open('configuration.json'))  # New variable example
        self.current_time=time.time()
        self.ist_timezone=pytz.timezone('Asia/Kolkata')
        self.reset_tracker=0
        self.reset_threshold=None
        self.stop_thread=True

        # Method timing control - only for specific methods
        self.method_timing = {}
        self.method_intervals = {
            'workforce_efficiency': 3.0,  # Run every 3 seconds
            'desk_occupancy': 3.0,       # Run every 3 seconds
            'resource_utilization': 3.0,  # Run every 3 seconds
            'workplace_area_occupancy': 3.0,  # Run every 3 seconds
            'camera_tampering': 5.0, # Run every 10 secs
            'send_random_data': 20.0
        }

    def new_function(self):  # New function example

        return "The meaning of life is: "
    
    def create_result_analytics(self,timestamp,analytics_name,class_name,zone_name,type,value):
        message={
            "sensor_id":self.sensor_id,
            "datetimestamp":timestamp,
            "analytic_type":analytics_name,
            "class_name":class_name,
            "area":zone_name,
            "type":type,
            "value":value
        }
        if self.results_analytics_queue.full():
            # print(self.results_analytics_queue.get(),"Lost Data")
            self.results_analytics_queue.get()
            self.results_analytics_queue.put(message)
        else:
            self.results_analytics_queue.put(message)
    
    def encode_frame_to_base64(self):
        _, buffer = cv2.imencode('.jpg', self.image)
        jpeg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpeg_as_text
    
    def encode_frame_to_bytes(self,quality=95,anpr=None):
        if anpr is None:
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', image, encode_param)
        else:
            _, buffer = cv2.imencode('.png',anpr)
        return buffer.tobytes()
    
    

    def create_result_events(self,xywh,class_name,sub_category,parameters,datetimestamp_trackerid,confidence=1,anprimage=None):
        
        if anprimage is not None:
            anprimage = cv2.cvtColor(anprimage, cv2.COLOR_BGR2RGB)
            image=self.encode_frame_to_bytes(100,anprimage)
            height, width = anprimage.shape[:2]
            anpr_status="True"
        else:
            image=self.encode_frame_to_bytes()
            height, width = self.image.shape[:2]
            anpr_status="False"
        
        message={
            "sensor_id":self.sensor_id,
            "org_img":image,
            "absolute_bbox":[{"xywh":xywh,"class_name":class_name,"confidence":confidence,"subcategory":sub_category,"parameters":parameters,"anpr":anpr_status}],
            "datetimestamp_trackerid":datetimestamp_trackerid,
            "imgsz":f"{width}:{height}",
            "color":"#FFFF00"
        }
        if self.results_events_queue.full():
            # print(self.results_events_queue.get(),"Lost Data")
            self.results_events_queue.put(message)
        else:
            self.results_events_queue.put(message)
    
    def is_person_in_zone(self, person_coordinates, zone_polygon):
        """
        Check if the center of the person's bounding box is inside a zone.
        person_coordinates: [xmin, ymin, xmax, ymax]
        zone_polygon: shapely Polygon object for the zone
        """
        x_center = int(person_coordinates[0] + person_coordinates[2]) / 2
        y_center = (person_coordinates[3])
        person_center = Point(x_center, y_center)
        return zone_polygon.contains(person_center)
    
    def is_object_in_zone(self, person_coordinates, zone_polygon):
        """
        Check if the center of the person's bounding box is inside a zone.
        person_coordinates: [xmin, ymin, xmax, ymax]
        zone_polygon: shapely Polygon object for the zone
        """
        x_center = (person_coordinates[0] + person_coordinates[2]) / 2
        y_center = (person_coordinates[1] + person_coordinates[3]) / 2
        person_center = Point(x_center, y_center)
        return zone_polygon.contains(person_center)
    
    def find_clusters_by_zone(self,gathered_tracker_ids):
        parent = {}

        # Find function with path compression
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        # Union function
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Initialize Union-Find structure and process pairs
        zone_clusters = {}
        for zone_name, pairs in gathered_tracker_ids.items():
            for tracker1, tracker2 in pairs:
                if tracker1 not in parent:
                    parent[tracker1] = tracker1
                if tracker2 not in parent:
                    parent[tracker2] = tracker2
                union(tracker1, tracker2)

        # Group elements by their root for each zone
        for tracker in parent:
            root = find(tracker)
            zone_clusters.setdefault(root, []).append(tracker)

        # Organize clusters by zone name
        result = {}
        for zone_name in gathered_tracker_ids:
            zone_result = []
            seen = set()
            for tracker1, tracker2 in gathered_tracker_ids[zone_name]:
                root1, root2 = find(tracker1), find(tracker2)
                if root1 not in seen:
                    zone_result.append(zone_clusters[root1])
                    seen.add(root1)
                if root2 not in seen:
                    zone_result.append(zone_clusters[root2])
                    seen.add(root2)
            result[zone_name] = zone_result

        return result

    def cleaning_list_with_last_frames(self,list_of_ids):
        results=[]
        for i in list_of_ids:
            if i in self.last_n_frame_tracker_ids:
                results.append(i)
        return results
    
    def send_random_data(self):
        xywh=[0,0,100,100]
        datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
        subcategory="Unsafe"
        self.create_result_events(xywh,"Act",f"Unsafe Activity-{subcategory}",{},datetimestamp_trackerid,confidence=0.6)
        #print("Done sir")

    def cleaning_events_data_with_last_frames(self):
        logging.info("cleaning_started")

        # Cleaning Violations
        for activity in self.active_activities_for_cleaning:
            if activity == "people_gathering":
                self.violation_id_data[activity]=[ tracker_id for tracker_id in self.violation_id_data[activity] if tracker_id in self.last_n_frame_tracker_ids]
            
            elif activity=="traffic_overspeeding":
                self.violation_id_data[activity]=[ tracker_id for tracker_id in self.violation_id_data[activity] if tracker_id in self.last_n_frame_tracker_ids]
                for tracker_id in list(self.traffic_overspeeding_speed_data.keys()):
                    # Check if tracker_id is not in self.last_n_frame_tracker_ids
                    if tracker_id not in self.last_n_frame_tracker_ids:
                        # Remove the tracker_id from self.traffic_overspeeding_distancewise_data
                        del self.traffic_overspeeding_speed_data[tracker_id]
            
            elif activity=="traffic_overspeeding_distancewise":
                for tracker_id in list(self.traffic_overspeeding_distancewise_data.keys()):
                    # Check if tracker_id is not in self.last_n_frame_tracker_ids
                    if tracker_id not in self.last_n_frame_tracker_ids:
                        # Remove the tracker_id from self.traffic_overspeeding_distancewise_data
                        del self.traffic_overspeeding_distancewise_data[tracker_id]
            
            elif activity=="ppe":
                for acts in self.parameters_data[activity]["subcategory_mapping"]:
                    self.violation_id_data[activity][acts]=[ tracker_id for tracker_id in self.violation_id_data[activity][acts] if tracker_id in self.last_n_frame_tracker_ids]
                for zone_name in self.zone_data[activity].keys():
                    self.ppe_data[zone_name] = {
                        key: value
                        for key, value in self.ppe_data[zone_name].items()
                        if key in self.last_n_frame_tracker_ids
                    }
            
            elif activity=="perimeter_monitoring":
                self.violation_id_data[activity]=[ tracker_id for tracker_id in self.violation_id_data[activity] if tracker_id in self.last_n_frame_tracker_ids]
                for zone_name in self.zone_data[activity].keys():
                    self.perimeter_monitoring_data[zone_name] = {
                        key: value
                        for key, value in self.perimeter_monitoring_data[zone_name].items()
                        if key in self.last_n_frame_tracker_ids
                    }

            elif activity=="climbing" or activity=="time_based_unauthorized_access"  :
                self.violation_id_data[activity]=[ tracker_id for tracker_id in self.violation_id_data[activity] if tracker_id in self.last_n_frame_tracker_ids]
                for zone_name in self.zone_data[activity].keys():
                    self.climbing_data[zone_name] = {
                        key: value
                        for key, value in self.climbing_data[zone_name].items()
                        if key in self.last_n_frame_tracker_ids
                    }
        
            elif activity=="vehicle_interaction":
                self.violation_id_data[activity]=[ tracker_id for tracker_id in self.violation_id_data[activity] if tracker_id in self.last_n_frame_tracker_ids]
                for zone_name in self.zone_data[activity].keys():
                    self.vehicle_interaction_data[zone_name] = {
                        key: value
                        for key, value in self.vehicle_interaction_data[zone_name].items()
                        if key in self.last_n_frame_tracker_ids
                    }
        
            elif activity=="person_violations":
                for acts in self.parameters_data[activity]["subcategory_mapping"]:
                    self.violation_id_data[activity][acts]=[ tracker_id for tracker_id in self.violation_id_data[activity][acts] if tracker_id in self.last_n_frame_tracker_ids]
                for zone_name in self.zone_data[activity].keys():
                    self.person_violations_data[zone_name] = {
                        key: value
                        for key, value in self.person_violations_data[zone_name].items()
                        if key in self.last_n_frame_tracker_ids
                    }

        logging.info("cleaning_completed")


    def reset_and_cleaning_variables(self):
        if self.workforce_efficiency_data["current_frame_count"] >= self.workforce_efficiency_data["send_data_frame_limit"]:
            # Reset counters for the next interval
            self.workforce_efficiency_data["current_frame_count"]=0  # Reset interval count
            self.workforce_efficiency_data["timestamp"] = datetime.now(self.ist_timezone).isoformat()  # Update timestamp
            self.workforce_efficiency_data["time"]=time.time()

    def xywh_original_percentage(self,box):
        min_x, min_y, max_x, max_y = box[0],box[1],box[2],box[3]
        xywh=[float(min_x*100/self.original_width),float(min_y*100/self.original_height),float((max_x-min_x)*100/self.original_width),float((max_y-min_y)*100/self.original_height)]
        return xywh

    def bottom_center(self,box):
        """
        Given a bounding box [xmin, ymin, xmax, ymax],
        return the bottom-center point (x, y) as integers.
        """
        x_center = int((box[0] + box[2]) / 2)
        y_bottom = int(box[3])
        return (x_center, y_bottom)

    def get_keypoints(self):
        """Get the COCO keypoints and their left/right flip coorespondence map."""
        keypoints = {
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
        return keypoints

    def track_person_movement(self,person_id, keypoints,zone_name):
        """Track if a person has moved for unconscious detection."""
        current_time = time.time()
        center_point = np.mean(keypoints[:, :2], axis=0)
        if person_id in self.person_violations_data[zone_name]:
            movement = np.linalg.norm(center_point - self.person_violations_data[zone_name][person_id]["last_pos"])
            if movement < 5:  # If movement is very small
                self.person_violations_data[zone_name][person_id]["unmoving_time"] += current_time - self.person_violations_data[zone_name][person_id]["last_time"]
            else:
                self.person_violations_data[zone_name][person_id]["unmoving_time"] = 0  # Reset timer if person moves

            self.person_violations_data[zone_name][person_id]["last_pos"]=center_point 
            self.person_violations_data[zone_name][person_id]["last_time"]=current_time
        else:
            self.person_violations_data[zone_name][person_id]={}
            self.person_violations_data[zone_name][person_id]["last_pos"]=center_point 
            self.person_violations_data[zone_name][person_id]["last_time"]=current_time
            self.person_violations_data[zone_name][person_id]["unmoving_time"] = 0
            self.person_violations_data[zone_name][person_id]["hands_above_start"]=current_time
            self.person_violations_data[zone_name][person_id]["fallen"]=0

        return self.person_violations_data[zone_name][person_id]["unmoving_time"]

    def merge_box(self,box1,box2):
        # Compute the union of the two bounding boxes
        xmin = min(box1[0], box2[0])
        ymin = min(box1[1], box2[1])
        xmax = max(box1[2], box2[2])
        ymax = max(box1[3], box2[3])
        union_box = [xmin, ymin, xmax, ymax]
        return union_box

    def calculate_iou(boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        
        Each box should be a list or tuple in the format:
        [xmin, ymin, xmax, ymax]
        """
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of the intersection rectangle
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight
        
        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the Intersection over Union by taking the intersection
        # area and dividing it by the sum of the prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return iou
    
    def calculate_euclidean_distance(self,coord_start, coord_end):
        # Convert coordinates to numpy arrays for vectorized operations
        coord_start = np.array(coord_start)
        coord_end = np.array(coord_end)
        
        # Calculate the Euclidean distance
        distance = np.linalg.norm(coord_end - coord_start)
        
        return distance
        
    # Function to calculate the distance between two points
    def calculate_distance(self,x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Function to project a point onto a line segment and return the projection point
    def project_point_on_line(self,x1, y1, x2, y2, px, py):
        # Vector from point1 to point2
        dx, dy = x2 - x1, y2 - y1
        # Vector from point1 to point P
        px1, py1 = px - x1, py - y1
        # Project point P onto the line defined by (x1, y1) -> (x2, y2)
        t = (px1 * dx + py1 * dy) / (dx * dx + dy * dy)
        # Get the projected point on the line
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return proj_x, proj_y

    # Function to calculate the closest line's projected distance from the midpoint of the vehicle's path
    def closest_line_projected_distance(self,vehicle_path, lines):
        [(x1, y1), (x2, y2)] = vehicle_path  # Vehicle path coordinates
        # print(self.calculate_distance(x1,y1,x2,y2))
        
        # Calculate the midpoint of the vehicle's path
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        
        min_distance = float('inf')
        close_line=None
        total_distance_travelled=None
        travelled_distance=None
        closest_lane=None
        
        # Loop through each line to calculate the projected distance from the midpoint of the vehicle path
        for lane_name, line in lines.items():
            [[x1_line, y1_line], [x2_line, y2_line]] = line  # Line coordinates
            
            # Project the midpoint of the vehicle path onto the line
            proj_mid_x, proj_mid_y = self.project_point_on_line(x1_line, y1_line, x2_line, y2_line, midpoint_x, midpoint_y)
            
            # Calculate the distance between the midpoint and the projected point on the line
            distance = self.calculate_distance(midpoint_x, midpoint_y, proj_mid_x, proj_mid_y)
            
            # Track the minimum distance (closest line's distance)
            if distance < min_distance:
                min_distance = distance
                close_line= line
                closest_lane= lane_name
        
        if close_line is not None:
            [[x1_line, y1_line], [x2_line, y2_line]] = close_line
            proj_x1, proj_y1 = self.project_point_on_line(x1_line, y1_line, x2_line, y2_line, x1, y1)
            proj_x2, proj_y2 = self.project_point_on_line(x1_line, y1_line, x2_line, y2_line, x2, y2)
            travelled_distance = self.calculate_distance(proj_x1, proj_y1 ,proj_x2, proj_y2)
            total_distance_travelled= self.calculate_distance(x1, y1 ,x2, y2)
        
        return travelled_distance, total_distance_travelled,closest_lane

    def crop_image_numpy(self,image_array, bounding_box):
        """
        Crops the image represented as a numpy array based on the provided bounding box,
        and returns both the cropped image and the new bounding box relative to the cropped image.

        :param image_array: The image as a numpy array.
        :param bounding_box: A tuple of (left, upper, right, lower) coordinates.
        :return: A tuple (cropped_image, new_bounding_box)
        """
        # Extract the coordinates from the bounding box
        left, upper, right, lower = bounding_box
        
        # Crop the image using array slicing
        cropped_image = image_array[int(upper+20):int(lower+20), int(left+20):int(right+20)]
        
        
        return cropped_image

    def workforce_efficiency(self):
        
        person_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["workforce_efficiency"]["subcategory_mapping"]]
        count={}
        for idx in person_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["workforce_efficiency"].items():
                if self.is_person_in_zone(box, zone_polygon):
                    if zone_name not in count:
                        count[zone_name] = 1  # Increment count for the zone
                    else:
                        count[zone_name] +=1
                    
        for zone_name in self.zone_data["workforce_efficiency"].keys():
            if zone_name in count:
                self.workforce_efficiency_data[zone_name]=max(self.workforce_efficiency_data[zone_name],count[zone_name])
        
        #Increment the number of current frame count
        self.workforce_efficiency_data["current_frame_count"] += 1  # Increment interval count
        # print(self.workforce_efficiency_data["current_frame_count"])
        #print (self.workforce_efficiency_data)
        # Check if the current interval (self.workforce_efficiency_data[1]) exceeds the set time threshold
        if self.workforce_efficiency_data["current_frame_count"] >= self.workforce_efficiency_data["send_data_frame_limit"]:
            for zone_name in self.zone_data["workforce_efficiency"].keys():
                # Calculate efficiency result based on the accumulated person count
                result = float(self.workforce_efficiency_data[zone_name])
                self.workforce_efficiency_data[zone_name]=0
                # Create and store the result with the timestamp
                if zone_name == "Complete_zone":
                    self.create_result_analytics(self.workforce_efficiency_data["timestamp"],"WORKPLACE_EFFICIENCY","person",zone_name,"AVERAGE",result)
                else:
                    self.create_result_analytics(self.workforce_efficiency_data["timestamp"],"ZONE_EFFICIENCY","person",zone_name,"AVERAGE",result)
            
        #print("Workforce Efficiency",self.workforce_efficiency_data)
    
    def desk_occupancy(self):
        
        person_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["desk_occupancy"]["subcategory_mapping"]]
        count={}
        # Iterate over all detected objects
        for idx in person_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["desk_occupancy"].items():
                if self.is_person_in_zone(box, zone_polygon):  # Only check for 'person' class
                    if zone_name not in count:
                        count[zone_name] = 1  # Increment count for the zone
                            
        for zone_name in self.zone_data["desk_occupancy"].keys():
            if zone_name in count:
                self.desk_occupancy_data[zone_name]=max(self.desk_occupancy_data[zone_name],count[zone_name])
        #print (self.desk_occupancy_data)
        # Check if the current interval (self.workforce_efficiency_data[1]) exceeds the set time threshold
        if self.workforce_efficiency_data["current_frame_count"] >= self.workforce_efficiency_data["send_data_frame_limit"]:
            for zone_name in self.zone_data["desk_occupancy"].keys():
                # Calculate efficiency result based on the accumulated person count
                result = float(self.desk_occupancy_data[zone_name])
                self.desk_occupancy_data[zone_name]=0
                # Create and store the result with the timestamp
                self.create_result_analytics(self.workforce_efficiency_data["timestamp"],"DESK_OCCUPANCY","person",zone_name,"AVERAGE_PERCENTAGE",result*100)
        #print("Desk Occupancy",self.desk_occupancy_data)

    def resource_utilization(self):
        
        person_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["resource_utilization"]["subcategory_mapping"]]
        count={}
        # Iterate over all detected objects
        for idx in person_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["resource_utilization"].items():
                if self.is_person_in_zone(box, zone_polygon):  # Only check for 'person' class
                    if zone_name not in count:
                        count[zone_name] = 1  # Increment count for the zone
                            
        for zone_name in self.zone_data["resource_utilization"].keys():
            if zone_name in count:
                self.resource_utilization_data[zone_name]=max(self.resource_utilization_data[zone_name],count[zone_name])
        
        #print (self.resource_utilization_data)
        # Check if the current interval (self.workforce_efficiency_data[1]) exceeds the set time threshold
        if self.workforce_efficiency_data["current_frame_count"] >= self.workforce_efficiency_data["send_data_frame_limit"]:
            for zone_name in self.zone_data["resource_utilization"].keys():
                # Calculate efficiency result based on the accumulated person count
                result = float(self.resource_utilization_data[zone_name])
                self.resource_utilization_data[zone_name]=0
                # Create and store the result with the timestamp
                self.create_result_analytics(self.workforce_efficiency_data["timestamp"],"RESOURCE_UTILIZATION","person",zone_name,"AVERAGE_PERCENTAGE",result*100)
        # print("Resource Utilization",self.resource_utilization_data)

    def entry_exit_WLE_logs(self):
        # Iterate over all detected objects
        for zone_name, zone_polygon in self.zone_data["entry_exit_WLE_logs"].items():
            for idx, (box, obj_class,tracker_id) in enumerate(zip(self.detection_boxes, self.classes,self.tracker_ids)):
                if obj_class in self.parameters_data["entry_exit_WLE_logs"]["subcategory_mapping"] and tracker_id not in self.entry_exit_WLE_logs_data[zone_name]["entry_ids"][obj_class]:
                    if self.is_person_in_zone(box, zone_polygon[0]): # Entry Check
                        self.entry_exit_WLE_logs_data[zone_name]["entry_ids"][obj_class].append(tracker_id)
                        if tracker_id in self.entry_exit_WLE_logs_data[zone_name]["exit_ids"][obj_class]:
                            if self.parameters_data["entry_exit_WLE_logs"]["entry_exit"]=="True":
                                self.entry_exit_WLE_logs_data[zone_name]["entry_count"][obj_class]+=1
                                self.entry_exit_WLE_logs_data[zone_name]["exit_ids"][obj_class].remove(tracker_id)
                            
                            if self.parameters_data["entry_exit_WLE_logs"]["wrong_lane"]=="True" and obj_class != "person":
                                anprimage=self.crop_image_numpy(self.image,box)
                                xywh=[0,0,100,100]
                                datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                self.create_result_events(xywh,obj_class,"Machine Vehicle Control-Wrong Lane",{},datetimestamp_trackerid,1,anprimage)
                                # Create Wrong Lane
                            # print("entry",self.entry_exit_logs_data[zone_name]["entry_count"])

                if obj_class in self.parameters_data["entry_exit_WLE_logs"]["subcategory_mapping"] and tracker_id not in self.entry_exit_WLE_logs_data[zone_name]["exit_ids"][obj_class]:
                    if self.is_person_in_zone(box, zone_polygon[1]): # Exit Check
                        self.entry_exit_WLE_logs_data[zone_name]["exit_ids"][obj_class].append(tracker_id) 
                        if tracker_id in self.entry_exit_WLE_logs_data[zone_name]["entry_ids"][obj_class]:
                            if self.parameters_data["entry_exit_WLE_logs"]["entry_exit"]=="True":
                                self.entry_exit_WLE_logs_data[zone_name]["exit_count"][obj_class]+=1
                                self.entry_exit_WLE_logs_data[zone_name]["entry_ids"][obj_class].remove(tracker_id)
                            # print("exit",self.entry_exit_logs_data[zone_name]["exit_count"])
        
        # Check if the current interval (self.workforce_efficiency_data[1]) exceeds the set time threshold
        if self.parameters_data["entry_exit_WLE_logs"]["entry_exit"]=="True" and (time.time() - self.parameters_data["entry_exit_WLE_logs"]["last_frame_time"]) >= self.parameters_data["entry_exit_WLE_logs"]["frame_send_sec"]:
            self.parameters_data["entry_exit_WLE_logs"]["last_frame_time"]=time.time()
            for zone_name in self.zone_data["entry_exit_WLE_logs"].keys():
                # Calculate efficiency result based on the accumulated person count
                for object_class in self.parameters_data["entry_exit_WLE_logs"]["subcategory_mapping"]:
                    result_entry = self.entry_exit_WLE_logs_data[zone_name]["entry_count"][object_class]
                    result_exit = self.entry_exit_WLE_logs_data[zone_name]["exit_count"][object_class]
                    timestamp=datetime.now(self.ist_timezone).isoformat()
                    #print(result_entry,result_exit)
                    
                    
                    if result_entry !=0:
                        self.entry_exit_WLE_logs_data[zone_name]["entry_count"][object_class]=0
                        self.create_result_analytics(timestamp,"ENTRY","object_class",zone_name,"ABSOLUTE",result_entry)
                    if result_exit !=0:
                        self.entry_exit_WLE_logs_data[zone_name]["exit_count"][object_class]=0
                        self.create_result_analytics(timestamp,"EXIT","object_class",zone_name,"ABSOLUTE",result_exit)
                        
                    self.entry_exit_WLE_logs_data[zone_name]["entry_ids"][object_class]=self.cleaning_list_with_last_frames(self.entry_exit_WLE_logs_data[zone_name]["entry_ids"][object_class])
                    self.entry_exit_WLE_logs_data[zone_name]["exit_ids"][object_class]=self.cleaning_list_with_last_frames(self.entry_exit_WLE_logs_data[zone_name]["exit_ids"][object_class])
        #print("Entry_Exit_Logs",self.entry_exit_WLE_logs_data) 

    def workplace_area_occupancy(self):
        person_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["workplace_area_occupancy"]["subcategory_mapping"]]
        count={}
        
        for idx in person_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["workplace_area_occupancy"].items():
                if self.is_person_in_zone(box, zone_polygon):
                    if zone_name not in count:
                        count[zone_name] = 1  # Increment count for the zone
                    else:
                        count[zone_name] +=1
                    
        for zone_name in self.zone_data["workplace_area_occupancy"].keys():
            if zone_name in count:
                self.workplace_area_occupancy_data[zone_name]["occupancy"]=max(self.workplace_area_occupancy_data[zone_name]["occupancy"],count[zone_name])
        #print(self.workplace_area_occupancy_data)
        # Check if the current interval (self.workforce_efficiency_data[1]) exceeds the set time threshold
        if self.workforce_efficiency_data["current_frame_count"] >= self.workforce_efficiency_data["send_data_frame_limit"]:
            for zone_name in self.zone_data["workplace_area_occupancy"].keys():
                # Calculate efficiency result based on the accumulated person count
                result = float((self.workplace_area_occupancy_data[zone_name]["occupancy"])/self.workplace_area_occupancy_data[zone_name]["capacity"])
                self.workplace_area_occupancy_data[zone_name]["occupancy"]=0
                # Create and store the result with the timestamp
                self.create_result_analytics(self.workforce_efficiency_data["timestamp"],"WORKPLACE_AREA_OCCUPANCY","person",zone_name,"AVERAGE_PERCENTAGE",result*100)
       
        #print("Workforce Occupancy",self.workplace_area_occupancy_data)

    def people_gathering(self):
        gathered_tracker_ids=defaultdict(list)
        for zone_name, zone_polygon in self.zone_data["people_gathering"].items():
            gathered_tracker_ids[zone_name]=[]
        # print(self.detection_boxes,self.classes)
        for (box, obj_class, tracker_id), (boxy, obj_classy, tracker_idy) in combinations(zip(self.detection_boxes, self.classes, self.tracker_ids), 2):
                if obj_class != "person" or obj_classy != "person":  
                    continue  # Avoid duplicate and self-comparison

                # Compute Euclidean distance
                dist = distance.euclidean(self.bottom_center(box), self.bottom_center(boxy))
                threshold=max((box[2] - box[0]), (boxy[2] - boxy[0])) * 1.35
                if dist < threshold:
                    for zone_name, zone_polygon in self.zone_data["people_gathering"].items():
                        if self.is_object_in_zone(box, zone_polygon) and self.is_person_in_zone(boxy,zone_polygon):
                            gathered_tracker_ids[zone_name].append([tracker_id,tracker_idy])
                            break
        
        clusters = self.find_clusters_by_zone(gathered_tracker_ids)
        active_clusters = {} 
        for zone_name, zone_clusters in clusters.items():
            active_clusters[zone_name]=[]
            for cluster in zone_clusters:
                if len(cluster) >= self.parameters_data["people_gathering"]["person_limit"]:
                    prev_violation_count=0
                    for person_id in cluster:
                        if person_id in self.violation_id_data["people_gathering"]:
                            prev_violation_count+=1
                        
                    if len(cluster) > 0 and int((prev_violation_count * 100) / len(cluster)) < 70:
                        active_clusters[zone_name].append(cluster)
                        logging.info(active_clusters[zone_name])
            
            if len(active_clusters[zone_name])>0:
                self.people_gathering_data[zone_name]["frame_accuracy"]+=1
                self.people_gathering_data[zone_name]["last_time"]=time.time()

            if self.people_gathering_data[zone_name]["frame_accuracy"]>self.parameters_data["people_gathering"]["frame_accuracy"]:
                for clust in active_clusters[zone_name]:
                    gathered_box=None
                    for idx, (box, obj_class, tracker_id) in enumerate(zip(self.detection_boxes, self.classes, self.tracker_ids)):
                        if obj_class == "person" and tracker_id in clust:
                            if gathered_box is None or not gathered_box.any():
                                gathered_box = np.copy(box)
                            else:
                                gathered_box = self.merge_box(gathered_box, box)
                            logging.info(gathered_box)
                            self.violation_id_data["people_gathering"].append(tracker_id)

                    
                    xywh=self.xywh_original_percentage(gathered_box)
                    datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                    self.create_result_events(xywh,"person","Behavioural Safety-people_gathering",{"person_count":len(clust)},datetimestamp_trackerid,confidence=1)

                    self.people_gathering_data[zone_name]["frame_accuracy"]=0
                    self.people_gathering_data[zone_name]["last_time"]=time.time()

            elif(time.time()-self.people_gathering_data[zone_name]["last_time"])>self.parameters_data["people_gathering"]["last_time"]:
                self.people_gathering_data[zone_name]["frame_accuracy"]=0
                self.people_gathering_data[zone_name]["last_time"]=time.time()


    def traffic_overspeeding(self):
        
        for tracker_id, anchor in zip(self.tracker_ids, self.anchor_points):
            self.traffic_overspeeding_data[tracker_id].append(anchor)
        # print(self.traffic_overspeeding_data)
        # print(self.anchor_points)
        for idx, (box, obj_class,tracker_id) in enumerate(zip(self.detection_boxes, self.classes,self.tracker_ids)):
            if obj_class in self.parameters_data["traffic_overspeeding"]["speed_limit"].keys():
                for zone_name, zone_polygon in self.zone_data["traffic_overspeeding"].items():
                    if self.is_person_in_zone(box, zone_polygon) and tracker_id not in self.violation_id_data["traffic_overspeeding"]:
                        if len(self.traffic_overspeeding_data[tracker_id]) == self.parameters_data["traffic_overspeeding"]["frame_to_track"]:
                            #print(self.traffic_overspeeding_data[tracker_id])
                            #print(self.time_stamp)
                            coordinate_start = self.traffic_overspeeding_data[tracker_id][-1]
                            coordinate_end = self.traffic_overspeeding_data[tracker_id][0]
                            distance = self.calculate_euclidean_distance(coordinate_start, coordinate_end)
                            time = float(self.time_stamp[-1]-self.time_stamp[0])
                            speed = float(distance / time * 3.6)*self.parameters_data["traffic_overspeeding"]["calibration"] # km/h
                            if speed > 10:
                                self.traffic_overspeeding_speed_data[tracker_id].add(speed)
                            # Check if the list length is odd
                            if len(self.traffic_overspeeding_speed_data[tracker_id]) % 2 != 0:
                                # Get the median (middle element for odd-length list)
                                median_speed = self.traffic_overspeeding_speed_data[tracker_id][len(self.traffic_overspeeding_speed_data[tracker_id]) // 2]
                                #print(f"Speed {obj_class} {tracker_id} {median_speed} {speed}")  # Display the median and current speed
                            
                                if median_speed > self.parameters_data["traffic_overspeeding"]["speed_limit"][obj_class] and self.is_person_in_zone(box,self.anprscan_line_area):
                                    # Get the bounding rectangle of the polygon
                                    self.violation_id_data["traffic_overspeeding"].append(tracker_id)
                                    anprimage=self.crop_image_numpy(self.image,box)
                                    xywh=[0,0,100,100]
                                    datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                    self.create_result_events(xywh,obj_class,"Machine Vehicle Control-Overspeeding",{"speed":median_speed},datetimestamp_trackerid,1,anprimage)
                                
        # print("Traffic_Overspeeding_Running")

    def traffic_overspeeding_distancewise(self):
        
        # print(self.traffic_overspeeding_distancewise_data)
        # print(self.anchor_points_original)
        # print(self.distancewise_tracking)
        #print(self.zone_data)
        for idx, (box, obj_class,tracker_id,anchor) in enumerate(zip(self.detection_boxes, self.classes,self.tracker_ids,self.anchor_points_original)):
            if obj_class in self.parameters_data["traffic_overspeeding_distancewise"]["speed_limit"].keys():
                for zone_name, zone_polygon in self.zone_data["traffic_overspeeding_distancewise"].items():
                    if self.is_person_in_zone(box, zone_polygon):
                        if tracker_id not in self.distancewise_tracking and tracker_id not in self.violation_id_data["traffic_overspeeding_distancewise"]:
                                #print(self.traffic_overspeeding_data[tracker_id])
                                #print(self.time_stamp)
                                self.distancewise_tracking.append(tracker_id)
                                self.traffic_overspeeding_distancewise_data[tracker_id]["entry_anchor"]=anchor
                                self.traffic_overspeeding_distancewise_data[tracker_id]["entry_time"]=self.time_stamp[-1]
                            
                    elif tracker_id in self.distancewise_tracking and tracker_id not in self.violation_id_data["traffic_overspeeding_distancewise"]:
                        vehicle_path=[anchor,self.traffic_overspeeding_distancewise_data[tracker_id]["entry_anchor"]]
                        projected_distance,total_distance,lane_name = self.closest_line_projected_distance(vehicle_path,self.parameters_data["traffic_overspeeding_distancewise"]["lines"])
                        distance=float(self.parameters_data["traffic_overspeeding_distancewise"]["real_distance"]*projected_distance/self.parameters_data["traffic_overspeeding_distancewise"]["lines_length"][lane_name])
                        speed=float(distance*3.6/(self.time_stamp[-1]-self.traffic_overspeeding_distancewise_data[tracker_id]["entry_time"]))*self.parameters_data["traffic_overspeeding_distancewise"]["calibration"]
                        #print(speed)
                        self.distancewise_tracking.remove(tracker_id)
                        if speed > self.parameters_data["traffic_overspeeding_distancewise"]["speed_limit"][obj_class]:
                            #print(tracker_id,obj_class,speed)
                            # Get the bounding rectangle of the polygon
                            self.violation_id_data["traffic_overspeeding_distancewise"].append(tracker_id)
                            anprimage=self.crop_image_numpy(self.image,box)
                            xywh=[0,0,100,100]
                            datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                            self.create_result_events(xywh,obj_class,"Machine Vehicle Control-Overspeeding",{"speed":speed},datetimestamp_trackerid,1,anprimage)

    def ppe(self):
        processed=[]
        # print("PPE",self.classes)
        ppe_objects=self.parameters_data["ppe"]["subcategory_mapping"]
        person_indices = [i for i, cls in enumerate(self.classes) if cls == "person"]
        ppe_indices = [i for i, cls in enumerate(self.classes) if cls in ppe_objects.keys()]
        # print("PPE_Data",self.ppe_data)
        # print("ppe",person_indices,ppe_indices)
        for idx in person_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            #print(tracker_id)
            # print(self.zone_data["ppe"])
            for zone_name, zone_polygon in self.zone_data["ppe"].items():
                # print(self.is_person_in_zone(box, zone_polygon))
                # print("Violation_done",self.violation_id_data["ppe"])
                if tracker_id not in processed and self.is_object_in_zone(box, zone_polygon):
                    processed.append(tracker_id)
                    person_poly = Polygon([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]),  (box[2], box[1])])
                    for ppe_idx in ppe_indices:
                        ppe_box=self.detection_boxes[ppe_idx]
                        ppe_obj_class=self.classes[ppe_idx]
                        if self.is_object_in_zone(ppe_box,person_poly) and tracker_id not in self.violation_id_data["ppe"][ppe_obj_class]:
                            if tracker_id not in self.ppe_data[zone_name]:
                                self.ppe_data[zone_name][tracker_id] = {}
                            # Now, check if the ppe_obj_class key exists for this tracker_id
                            if ppe_obj_class not in self.ppe_data[zone_name][tracker_id]:
                                self.ppe_data[zone_name][tracker_id][ppe_obj_class] = 1
                            else:
                                self.ppe_data[zone_name][tracker_id][ppe_obj_class] += 1

                            
                            if self.ppe_data[zone_name][tracker_id][ppe_obj_class] > self.parameters_data["ppe"]["frame_accuracy"]:
                                self.violation_id_data["ppe"][ppe_obj_class].append(tracker_id)
                                xywh=self.xywh_original_percentage(box)
                                datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                subcategory=ppe_objects[ppe_obj_class]
                                self.create_result_events(xywh,obj_class,f"PPE-{subcategory}",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)
        # print("PPE_Running")


    def perimeter_monitoring(self):
        processed=[]

        offender_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["perimeter_monitoring"]["subcategory_mapping"]]

        for idx in offender_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["perimeter_monitoring"].items():
                if tracker_id not in processed and tracker_id not in self.violation_id_data["perimeter_monitoring"]  and self.is_person_in_zone(box, zone_polygon):
                    processed.append(tracker_id)
                
                    if tracker_id not in self.perimeter_monitoring_data[zone_name]:
                        self.perimeter_monitoring_data[zone_name][tracker_id] = 1
                    else:
                        self.perimeter_monitoring_data[zone_name][tracker_id]+= 1

                    if self.perimeter_monitoring_data[zone_name][tracker_id] > self.parameters_data["perimeter_monitoring"]["frame_accuracy"]:
                        self.violation_id_data["perimeter_monitoring"].append(tracker_id)
                        xywh=self.xywh_original_percentage(box)
                        datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                        self.create_result_events(xywh,obj_class,f"Security-{ppe_obj_class}",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)
        # print("Perimeter_Monitoring_Running")

    def climbing(self):
        processed=[]

        offender_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["climbing"]["subcategory_mapping"]]

        for idx in offender_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["climbing"].items():
                if tracker_id not in processed and tracker_id not in self.violation_id_data["climbing"]  and self.is_person_in_zone(box, zone_polygon):
                    processed.append(tracker_id)
                
                    if tracker_id not in self.climbing_data[zone_name]:
                        self.climbing_data[zone_name][tracker_id] = 1
                    else:
                        self.climbing_data[zone_name][tracker_id]+= 1

                    if self.climbing_data[zone_name][tracker_id] > self.parameters_data["climbing"]["frame_accuracy"]:
                        self.violation_id_data["climbing"].append(tracker_id)
                        xywh=self.xywh_original_percentage(box)
                        datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                        self.create_result_events(xywh,obj_class,f"Hazardous Area-Climbing",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)
                        
    def time_based_unauthorized_access(self):
        if time.time()-self.parameters_data["time_based_unauthorized_access"]["last_check_time"]>1:
            self.parameters_data["time_based_unauthorized_access"]["last_check_time"]=time.time()
            # Loop through the scheduled times (you may have multiple schedules)
            for schedule in self.parameters_data["time_based_unauthorized_access"]["scheduled_time"]:
                # Get the time intervals (start_time, end_time) and days from the current schedule
                time_start_end = schedule["time_start_end"]
                days_of_week = schedule["days"]
                timezone_str = schedule["timezone"]
                # Set up the timezone from the provided string
                timezone = pytz.timezone(timezone_str)
                
                # Get current time in the specified timezone
                current_time = datetime.now(timezone).time()
                current_day = datetime.now(timezone).weekday()  # Monday = 0, Sunday = 6
                
                
                # Check if today is one of the specified days
                current_day_name = datetime.now(timezone).strftime('%A')  # Get the current day as a name (e.g., Monday)
                if current_day_name not in days_of_week:
                    return f"Today is not a valid day for the task. ({current_day_name} is not scheduled.)"
                
                # Loop through all time intervals in the 'time_start_end' list and check if the current time is within any of them
                for time_range in time_start_end:
                    start_time_str, end_time_str = time_range
                    
                    # Convert start and end time strings to time objects
                    start_time = datetime.strptime(start_time_str, "%H:%M").time()
                    end_time = datetime.strptime(end_time_str, "%H:%M").time()
                    
                    # Check if the current time is within the specified time range
                    if start_time <= current_time <= end_time:
                       
                        processed=[]

                        offender_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["time_based_unauthorized_access"]["subcategory_mapping"]]

                        for idx in offender_indices:
                            box=self.detection_boxes[idx]
                            obj_class=self.classes[idx]
                            tracker_id=self.tracker_ids[idx]
                            for zone_name, zone_polygon in self.zone_data["time_based_unauthorized_access"].items():
                                if tracker_id not in processed and tracker_id not in self.violation_id_data["time_based_unauthorized_access"]  and self.is_person_in_zone(box, zone_polygon):
                                    processed.append(tracker_id)
                                
                                    if tracker_id not in self.climbing_data[zone_name]:
                                        self.climbing_data[zone_name][tracker_id] = 1
                                    else:
                                        self.climbing_data[zone_name][tracker_id]+= 1

                                    if self.climbing_data[zone_name][tracker_id] > self.parameters_data["time_based_unauthorized_access"]["frame_accuracy"]:
                                        self.violation_id_data["time_based_unauthorized_access"].append(tracker_id)
                                        xywh=self.xywh_original_percentage(box)
                                        datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                        self.create_result_events(xywh,obj_class,f"Hazardous Area-Unauthorized Access",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)

    def vehicle_interaction(self):
        """
        Vehicle interaction detection optimized for safety monitoring.
        Detects when vehicles in motion approach people or other vehicles,
        regardless of whether the approached object is moving or stationary.
        """
        # Track objects by category
        vehicles = []      # Moving machinery, vehicles
        people = []        # Humans
        processed_ids = [] # Track processed IDs
        
        # Get all objects in monitored zones
        for idx, (box, obj_class, tracker_id) in enumerate(zip(self.detection_boxes, self.classes, self.tracker_ids)):
            if tracker_id in self.violation_id_data["vehicle_interaction"]:
                continue  # Skip already detected violations
                
            for zone_name, zone_polygon in self.zone_data["vehicle_interaction"].items():
                if self.is_person_in_zone(box, zone_polygon) and tracker_id not in processed_ids:
                    processed_ids.append(tracker_id)
                    
                    # Calculate object properties
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    
                    # For angled view, use a point between center and bottom for better perspective
                    anchor_point = (int((box[0] + box[2]) / 2), int(box[1] + height * 0.7))
                    
                    # Calculate motion vector if previous position exists
                    motion_vector = (0, 0)
                    in_motion = False
                    
                    if tracker_id in self.prev_pos:
                        prev_box = self.prev_pos[tracker_id]
                        prev_h = prev_box[3] - prev_box[1]
                        prev_anchor = (int((prev_box[0] + prev_box[2]) / 2), int(prev_box[1] + prev_h * 0.7))
                        
                        # Calculate displacement
                        dx = anchor_point[0] - prev_anchor[0]
                        dy = anchor_point[1] - prev_anchor[1]
                        displacement = math.hypot(dx, dy)
                        
                        # Check if object is in motion
                        if displacement > self.parameters_data["vehicle_interaction"]["motion_thr"]:
                            motion_vector = (dx, dy)
                            in_motion = True
                    
                    # Create object data dictionary
                    obj_data = {
                        "tracker_id": tracker_id,
                        "box": box,
                        "class": obj_class,
                        "zone": zone_name,
                        "anchor_point": anchor_point,
                        "width": width,
                        "height": height,
                        "motion_vector": motion_vector,
                        "in_motion": in_motion
                    }
                    
                    # Categorize object
                    if obj_class in self.parameters_data["vehicle_interaction"]["vehicle_classes"]:
                        vehicles.append(obj_data)
                    
                    if obj_class in self.parameters_data["vehicle_interaction"]["person_classes"]:
                        people.append(obj_data)
        
        # Process vehicle-person interactions
        for vehicle in vehicles:
            # Skip if vehicle is not in motion - vehicle must be moving to be a risk
            if not vehicle["in_motion"]:
                continue
            
            for person in people:
                # Skip if not in same zone
                if vehicle["zone"] != person["zone"]:
                    continue
                
                # Skip if already identified as violation
                if vehicle["tracker_id"] in self.violation_id_data["vehicle_interaction"] and \
                   person["tracker_id"] in self.violation_id_data["vehicle_interaction"]:
                    continue
                
                # Calculate distance between objects
                v_point = vehicle["anchor_point"]
                p_point = person["anchor_point"]
                
                # Calculate distance metrics
                horizontal_dist = abs(v_point[0] - p_point[0])
                vertical_dist = abs(v_point[1] - p_point[1])
                
                # Proximity thresholds adjusted for angled view
                prox_h_threshold = (vehicle["width"] / 2 + person["width"] / 2) * \
                                  (1 + self.parameters_data["vehicle_interaction"]["hor_interaction_percentage"])
                prox_v_threshold = max(vehicle["height"], person["height"]) * \
                                  self.parameters_data["vehicle_interaction"]["ver_interaction_percentage"]
                
                # For angled view, use a combined approach
                proximity_violation = (horizontal_dist < prox_h_threshold and 
                                      vertical_dist < prox_v_threshold)
                
                if proximity_violation:
                
                    # Check if vehicle is moving toward person
                    approaching = False
                    
                    mv = vehicle["motion_vector"]
                    
                    # Vector from vehicle to person
                    vp_vector = (p_point[0] - v_point[0], p_point[1] - v_point[1])
                    
                    # Calculate angle between motion and direction to person
                    mv_mag = math.hypot(mv[0], mv[1])
                    vp_mag = math.hypot(vp_vector[0], vp_vector[1])
                    
                    tracker_id_pair = (min(vehicle["tracker_id"], person["tracker_id"]), max(vehicle["tracker_id"], person["tracker_id"]))
                    current_distance = math.sqrt((v_point[0] - p_point[0])**2 + (v_point[1] - p_point[1])**2)

                    if tracker_id_pair in self.prev_distances.keys():
                        prev_distance = self.prev_distances[tracker_id_pair]
                        distance_change = prev_distance - current_distance
                        approaching = distance_change > 0
                        
                    else:
                        self.prev_distances[tracker_id_pair]=current_distance
                        continue
                    
                    self.prev_distances[tracker_id_pair] = current_distance
                    
                    # Detect violation if vehicle is approaching person and proximity violation exists
                    if approaching and proximity_violation:
                        # Mark both objects as violations
                        self.violation_id_data["vehicle_interaction"].append(vehicle["tracker_id"])
                        self.violation_id_data["vehicle_interaction"].append(person["tracker_id"])
                        
                        # Create merged bounding box
                        merged_box = self.merge_box(vehicle["box"], person["box"])
                        xywh = self.xywh_original_percentage(merged_box)
                        
                        # Determine violation type
                        violation_type = "Approaching" if not person["in_motion"] else "Interaction"
                        
                        # Create unique ID with timestamp
                        timestamp = datetime.now(self.ist_timezone).isoformat()
                        event_id = f"{timestamp}_{vehicle['tracker_id']}_{person['tracker_id']}"
                        
                        # Report the event
                        self.create_result_events(
                            xywh, 
                            person["class"], 
                            f"Machine Control Area-Vehicle {violation_type}: {vehicle['class']} approaching {person['class']}", 
                            {"zone_name": vehicle["zone"],
                            "violation_type": violation_type,
                            "person_moving": person["in_motion"]}, 
                            event_id, 
                            confidence=1
                        )
        # Process vehicle-vehicle interactions
        if len(vehicles) > 1:
            for i in range(len(vehicles)):
                v1 = vehicles[i]
                
                for j in range(len(vehicles)):
                    if i == j:  # Skip comparing vehicle with itself
                        continue
                    
                    v2 = vehicles[j]
                    
                    # Skip if neither vehicle is moving - at least one must be in motion
                    if not (v1["in_motion"] or v2["in_motion"]):
                        continue
                    
                    # Skip if not in same zone
                    if v1["zone"] != v2["zone"]:
                        continue
                    
                    # Skip if already identified as violation
                    if v1["tracker_id"] in self.violation_id_data["vehicle_interaction"] and \
                       v2["tracker_id"] in self.violation_id_data["vehicle_interaction"]:
                        continue
                    
                    # Calculate distance metrics
                    p1 = v1["anchor_point"]
                    p2 = v2["anchor_point"]
                    horizontal_dist = abs(p1[0] - p2[0])
                    vertical_dist = abs(p1[1] - p2[1])
                    
                    # Proximity thresholds for vehicle-vehicle interactions
                    prox_h_threshold = (v1["width"] / 2 + v2["width"] / 2) * \
                                      (1 + self.parameters_data["vehicle_interaction"].get("vv_hor_interaction_percentage", 0.2))
                    prox_v_threshold = max(v1["height"], v2["height"]) * \
                                      self.parameters_data["vehicle_interaction"].get("vv_ver_interaction_percentage", 0.2)
                    
                    # Check proximity violation
                    proximity_violation = horizontal_dist < prox_h_threshold and vertical_dist < prox_v_threshold

                    if proximity_violation:
                        # print(v1["tracker_id"],v2["tracker_id"],v1["in_motion"],v2["in_motion"],horizontal_dist,vertical_dist,prox_h_threshold,prox_v_threshold, proximity_violation)
                   

                        approaching = False

                        tracker_id_pair = (min(v1["tracker_id"], v2["tracker_id"]), max(v1["tracker_id"], v2["tracker_id"]))
                        current_distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                        if tracker_id_pair in self.prev_distances.keys():
                            prev_distance = self.prev_distances[tracker_id_pair]
                            distance_change = prev_distance - current_distance
                            approaching = distance_change > 0
                            
                        else:
                            self.prev_distances[tracker_id_pair] = current_distance
                            continue
                        
                        self.prev_distances[tracker_id_pair] = current_distance

                        
                        # Detect violation if approaching and proximity violation
                        if approaching and proximity_violation:
                            # Mark both vehicles as violations
                            self.violation_id_data["vehicle_interaction"].append(v1["tracker_id"])
                            self.violation_id_data["vehicle_interaction"].append(v2["tracker_id"])
                            
                            # Create merged bounding box
                            merged_box = self.merge_box(v1["box"], v2["box"])
                            xywh = self.xywh_original_percentage(merged_box)
                            
                            # Determine violation type based on motion state
                            if v1["in_motion"] and v2["in_motion"]:
                                violation_type = "Collision Risk"
                            else:
                                violation_type = "Approaching"
                            
                            # Create unique ID with timestamp
                            timestamp = datetime.now(self.ist_timezone).isoformat()
                            event_id = f"{timestamp}_{v1['tracker_id']}_{v2['tracker_id']}"
                            
                            # Report the event
                            self.create_result_events(
                                xywh, 
                                v2["class"], 
                                f"Machine Control Area-Vehicle {violation_type}: {v1['class']} approaching {v2['class']}",
                                {"zone_name": v1["zone"],
                                "violation_type": violation_type,
                                "both_moving": v1["in_motion"] and v2["in_motion"]}, 
                                event_id, 
                                confidence=1
                            )
        # print("Vehicle Interaction Running")

    def person_violations(self):

        processed=[]
        keypoints_index =self.keypoints_index
        if self.pose_detection_boxes is not None:
            for i, (person_id, person_box,person_keypoints) in enumerate(zip(self.pose_tracker_ids, self.pose_detection_boxes,self.pose_keypoints)):

                if person_keypoints is None:
                    # print(f"Skipping person {person_id} - keypoints are None")
                    continue
                for zone_name, zone_polygon in self.zone_data["person_violations"].items():     
                    if self.is_object_in_zone(person_box, zone_polygon) and person_id not in processed:
                        
                        # print(person_keypoints)
                        # Get KeyPoints
                        keypoints = np.array(person_keypoints)
                        # print(keypoints,keypoints.shape[0])
                        # Ensure we have enough keypoints for analysis
                        if keypoints.size == 0 or keypoints.shape[0] < 17:
                            continue  

                        if person_keypoints[keypoints_index["left_ankle"]][2]>0.30 or person_keypoints[int(keypoints_index["right_ankle"])][2]>0.30:
                                
                            # Bounding Box
                            x1, y1, x2, y2 = person_box
                            

                            left_shoulder = keypoints_index["left_shoulder"]
                            right_shoulder = keypoints_index["right_shoulder"]
                            left_hip = keypoints_index["left_hip"]
                            right_hip = keypoints_index["right_hip"]
                            left_knee = keypoints_index["left_knee"]
                            right_knee = keypoints_index["right_knee"]


                            # Compute Midpoints (Y-axis only for Sitting Detection)
                            mid_shoulder = np.mean(keypoints[[int(left_shoulder), int(right_shoulder)]], axis=0)  
                            mid_hip = np.mean(keypoints[[int(left_hip),int(right_hip)]], axis=0)      
                            mid_knee = np.mean(keypoints[[int(left_knee), int(right_knee)]], axis=0)   


                            # Compute Sitting Ratio (Only Y-Axis)
                            shoulder_hip_diff = abs(mid_shoulder[1] - mid_hip[1])
                            if shoulder_hip_diff > 0:  # Add zero-division check
                                sitting_ratio = abs(mid_hip[1] - mid_knee[1]) / shoulder_hip_diff
                            else:
                                # Skip this calculation if shoulder and hip are at the same Y position
                                continue
                            
                            # Compute Angles (Torso, Knee-Shoulder, Hip-Shoulder)
                            knee_shoulder_angle = abs(math.degrees(math.atan2(mid_shoulder[1] - mid_knee[1], 1)))  # Ignore X-axis
                            hip_shoulder_angle = abs(math.degrees(math.atan2(mid_shoulder[1] - mid_hip[1], 1)))  # Ignore X-axis
                            
                            # **Torso Angle Calculation** (Angle between mid-hip and mid-shoulder)
                            torso_vector = mid_shoulder - mid_hip
                            
                            # Compute angle relative to the vertical axis (Y-axis)
                            torso_angle = abs(math.degrees(math.atan2(torso_vector[1], torso_vector[0])))  # Swap X and Y
                        
                            # Convert angles > 90° to their correct counterpart
                            if torso_angle > 90:
                                torso_angle = 180 - torso_angle 
                            
                            # Idle time
                            unmoving_time = self.track_person_movement(person_id, keypoints,zone_name)

                            # Track fallen person frames
                            if torso_angle < self.parameters_data["person_violations"]["FALL_ANGLE_THRESHOLD"]:
                                # Now, check if the ppe_obj_class key exists for this tracker_id
                                self.person_violations_data[zone_name][person_id]["fallen"] +=1

                            else:
                                self.person_violations_data[zone_name][person_id]["fallen"]=1
                            
                            

                            if person_id not in self.violation_id_data["person_violations"]["unconscious"] and torso_angle < self.parameters_data["person_violations"]["FALL_ANGLE_THRESHOLD"] and unmoving_time > self.parameters_data["person_violations"]["UNCONSCIOUS_TIME_THRESHOLD"]:
                                self.violation_id_data["person_violations"]["unconscious"].append(person_id)
                                xywh=self.xywh_original_percentage(person_box)
                                datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                self.create_result_events(xywh,"person","Emergency Control-Person Unconcious",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)

                            # Detect Fallen (Torso is almost horizontal)
                            elif person_id not in self.violation_id_data["person_violations"]["fallen"] and self.person_violations_data[zone_name][person_id]["fallen"] >= self.parameters_data["person_violations"]["FALL_CONFIRMATION_FRAMES"]:
                                self.violation_id_data["person_violations"]["fallen"].append(person_id)
                                xywh=self.xywh_original_percentage(person_box)
                                datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                self.create_result_events(xywh,"person","Emergency Control-Person Fallen",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)

                            # Detect Sitting (Based on Y-ratio and Torso Angle)
                            elif person_id not in self.violation_id_data["person_violations"]["sitting"]  and sitting_ratio < self.parameters_data["person_violations"]["SEATED_RATIO_THRESHOLD"] and torso_angle > self.parameters_data["person_violations"]["SITTING_TORSO_ANGLE_THRESHOLD"] and "SITTING" in self.parameters_data["person_violations"]["acts"]:
                                self.violation_id_data["person_violations"]["sitting"].append(person_id)
                                xywh=self.xywh_original_percentage(person_box)
                                datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                                self.create_result_events(xywh,"person","Emergency Control-Person Sitting",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)

        # print("Person Violation Running")

    def fire_and_smoke(self):

        offender_indices = [i for i, cls in enumerate(self.classes) if cls in self.parameters_data["fire_and_smoke"]["subcategory_mapping"]]
        # print("Yes Baby",offender_indices)
        for idx in offender_indices:
            box=self.detection_boxes[idx]
            obj_class=self.classes[idx]
            tracker_id=self.tracker_ids[idx]
            for zone_name, zone_polygon in self.zone_data["fire_and_smoke"].items():
                # print(self.is_person_in_zone(box, zone_polygon))
                if self.is_person_in_zone(box, zone_polygon):
                    # print(self.fire_and_smoke_data[zone_name]["alert_interval"],self.fire_and_smoke_data[zone_name]["frame_count_last_time"],self.parameters_data["fire_and_smoke"]["last_frame_check"],self.fire_and_smoke_data[zone_name]["frame_count"],self.parameters_data["fire_and_smoke"]["alert_interval"])
                    if int((time.time() - self.fire_and_smoke_data[zone_name]["frame_count_last_time"])) >  self.parameters_data["fire_and_smoke"]["last_frame_check"]:
                        self.fire_and_smoke_data[zone_name]["frame_count"]=1
                        self.fire_and_smoke_data[zone_name]["frame_count_last_time"]=time.time()
                    else:
                        self.fire_and_smoke_data[zone_name]["frame_count"]+=1
                        self.fire_and_smoke_data[zone_name]["frame_count_last_time"]=time.time()

                    if self.fire_and_smoke_data[zone_name]["frame_count"] > self.parameters_data["fire_and_smoke"]["frame_accuracy"]:
                        # print("Fire Running",(time.time()-self.fire_and_smoke_data[zone_name]["alert_interval"]))
                        if int((time.time()-self.fire_and_smoke_data[zone_name]["alert_interval"])) > self.parameters_data["fire_and_smoke"]["alert_interval"] :
                            xywh=self.xywh_original_percentage(box)
                            datetimestamp_trackerid=f"{datetime.now(self.ist_timezone).isoformat()}"
                            self.create_result_events(xywh,obj_class,f"Emergency Control-{obj_classs}",{"zone_name":zone_name},datetimestamp_trackerid,confidence=1)
                            self.fire_and_smoke_data[zone_name]["frame_count"]=1
                            self.fire_and_smoke_data[zone_name]["alert_interval"]=time.time()
                            self.fire_and_smoke_data[zone_name]["frame_count_last_time"]=time.time()

        # print("Fire Smoke Running")

    def unattended_area(self):
        """
        Monitor for areas that should have someone present but don't.
        Alerts when no person is detected in a zone for too long.
        """
        current_time = time.time()
        
        # Check all detected objects for person presence in zones
        for idx, (box, obj_class) in enumerate(zip(self.detection_boxes, self.classes)):
            if obj_class in self.parameters_data["unattended_area"]["subcategory_mapping"]: 
                for zone_name, zone_polygon in self.zone_data["unattended_area"].items():
                    if self.is_object_in_zone(box, zone_polygon):
                        # Update person presence and reset timer
                        self.unattended_area_data[zone_name]["person_present"] = True
                        self.unattended_area_data[zone_name]["last_person_time"] = current_time
                        break  # Person found in a zone, no need to check other zones
        
        # Check for unattended zones and send alerts
        for zone_name,zone_polygon in self.zone_data["unattended_area"].items():
            if not self.unattended_area_data[zone_name]["person_present"]:
                # Check if zone has been unattended too long
                duration = current_time - self.unattended_area_data[zone_name]["last_person_time"]
                if (duration > self.parameters_data["unattended_area"]["time_threshold"]):
                    # Create violation event
                    self.unattended_area_data[zone_name]["last_person_time"] = current_time
                    datetimestamp = datetime.now(self.ist_timezone).isoformat()
                    minx, miny, maxx, maxy = zone_polygon.bounds
                    bbox = [int(minx), int(miny), int(maxx), int(maxy)]
                    self.create_result_events(
                        bbox,  # Full frame coordinates
                        "zone",
                        "Security-Unattended Area",
                        {
                            "zone_name": zone_name,
                            "duration": int(duration),
                            "status": "unattended"
                        },
                        datetimestamp,
                        confidence=1
                    )
                    
    def frame_hash(self,image):
        """Generate perceptual hash of the image to compare frames."""
        return imagehash.phash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    def camera_tampering(self):
        gray = cv2.cvtColor(self.model_image, cv2.COLOR_BGR2GRAY)
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
            brightness_change > self.parameters_data["camera_tampering"]["brightness_thres"] or
            variance < 15 or
            entropy_val < 1.5 or
            edge_ratio < 0.008
        )
        #print(brightness_change,variance,entropy_val,edge_ratio)

        # Initialize ref_hash if it has not been set (first-time setup or reset)
        if self.ref_hash is None:
            self.ref_hash = self.frame_hash(self.model_image)

        # Compare current frame to reference frame for camera movement detection
        if self.ref_hash is not None:
            current_hash = self.frame_hash(self.model_image)
            hash_diff = current_hash - self.ref_hash
        #print(self.ref_hash)
            

        # Check tampering condition based on time intervals and event triggering
        if int(time.time() - self.camera_tampering_data["alert_interval"]) > self.parameters_data["camera_tampering"]["alert_interval"]:
            if tampering or hash_diff > 20:
                self.camera_tampering_data["frame_count"] += 1

            if self.camera_tampering_data["frame_count"] > 0:
                xywh = [0, 0, 100, 100]
                datetimestamp_trackerid = f"{datetime.now(self.ist_timezone).isoformat()}"
                if hash_diff >20:
                    self.create_result_events(xywh, "Tampering", "Security-Camera_Movement", {}, datetimestamp_trackerid, confidence=1)
                    print("Work Done Movement")
                else:    
                    self.create_result_events(xywh, "Tampering", "Security-Camera_Tampering", {}, datetimestamp_trackerid, confidence=1)
                    print("Work Done Tampering")
                # Update the reference frame after tampering
                self.ref_hash = current_hash  # Update reference frame to the current one

                # Reset tampering data
                self.camera_tampering_data["frame_count"] = 0
                self.camera_tampering_data["alert_interval"] = time.time()
                self.prev_brightness = None


                

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# # -----------------------------------------------------------------------------------------------
# Defining Function for results
def setup_logging():
    current_dir = os.getcwd()
    root_dir = str(os.path.dirname(current_dir))
    LOG_DIR = root_dir+"/logs"
    LOG_FILE = os.path.join(LOG_DIR, 'Hailo.log')
    """Set up logging with rotation."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)  # Ensure log directory exists
    
    handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10 MB per log file, keep 5 backups
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

def result_to_tracker_OD(xyxy, confidence, class_id, class_name):
    if not xyxy:
        xyxy= np.empty((0,4))
    else: 
        xyxy=np.array(xyxy, dtype=float)
    confidence = np.array(confidence, dtype=float)
    class_id = np.array(class_id, dtype=int)
    data = {'class_name': np.array(class_name, dtype='<U20')} 
    mask = None  # Assuming masks are not used
    tracker_id = None  # Assuming tracking IDs are not used
    result = trackerinput(xyxy=xyxy,confidence=confidence,class_id=class_id,data=data)
    return result

def serialize_image(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, buffer =cv2.imencode('.jpeg',image,encode_param)
    img_base64=base64.b64encode(buffer).decode('utf-8')
    return img_base64

def create_message(img_base64,extra_data):
    # extra_data=json.dumps(extra_data)
    message = {
        'image':img_base64 ,
        'image_data':extra_data
    }
    return message

def json_format_serialization(final_results):
    final_json_results={}
    final_json_results["xyxy"]=final_results.xyxy.astype(int).tolist()
    final_json_results["confidence"]=final_results.confidence.tolist()
    final_json_results["class_id"]=final_results.class_id.tolist()
    final_json_results["tracker_id"]=final_results.tracker_id.tolist()
    final_json_results["class_name"]=final_results.data["class_name"].tolist() if "class_name" in final_results.data else []
    return final_json_results


def make_labelled_image(message):
    # Decode bytes → numpy image
    nparr = np.frombuffer(message["org_img"], np.uint8)
    org_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if org_img is None:
        raise ValueError("Could not decode org_img from bytes")

    # Parse image size (W:H)
    width, height = map(int, message["imgsz"].split(":"))
    #print(width,height)

    # Draw all bounding boxes
    for bbox in message["absolute_bbox"]:
        x_pct, y_pct, w_pct, h_pct = map(float, bbox["xywh"])
        
        #print(bbox["xywh"])

        # Convert percentages to pixel values
        x_center = int(x_pct * width/100)
        y_center = int(y_pct * height/100)
        w = int(w_pct * width/100)
        h = int(h_pct * height/100)

        # Convert (x,y,w,h) center format → top-left and bottom-right
        x1 = int(x_center)
        y1 = int(y_center)
        x2 = int(x_center + w)
        y2 = int(y_center + h)
        #print(x1,y1,x2,y2,"These are cordinate")

        # Convert HEX color → BGR
        color_hex = message["color"].lstrip("#")
        bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))

        # Draw bbox
        cv2.rectangle(org_img, (x1, y1), (x2, y2), bgr, 2)

        # Label text
        label = f"{bbox['class_name']} {bbox['confidence']:.2f}"
        cv2.putText(org_img, label, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

    # Encode back to bytes (JPEG)
    success, buffer = cv2.imencode(".jpg", org_img)
    if not success:
        raise ValueError("Could not encode labelled image")
    print( " Success Fully Labelled")
    return buffer.tobytes()

def upload_to_s3(S3_BUCKET_NAME,AWS_REGION, image_bytes, retries=3, delay=2):
    global s3_client,minio_client
    unique_filename = f"{uuid.uuid4()}.jpg"
    s3_url=None
    minio_url=None
    for attempt in range(retries):
        try:
            print("ATtempt Started")
            try:
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=unique_filename, Body=image_bytes, ContentType='image/jpg')
                print("uploaded successfully")
                s3_url= f"{unique_filename}"
            except:
                pass
            try:
                 minio_client.put_object(
                            Bucket="arrestominio",
                            Key=unique_filename,
                            Body=image_bytes,
                            ContentType='image/jpg'
                        )
                 minio_url = f"http://localhost:9000/arrestominio/{unique_filename}"
                 print(f"DEBUG: Successfully uploaded Image to {minio_url}")
            except:
                pass
                
            return s3_url,minio_url
            
        except Exception as e:
            logging.error(f"S3 upload failed, attempt {attempt + 1}/{retries}: {str(e)}")
            time.sleep(delay)
    logging.error("Failed to upload image to S3 after multiple retries.")
    
    return None


def send_data_to_kafka_broker(websocket_url=None,bootstrap_servers=None,ai_engine_pipeline=None,send_analytics_pipeline=None,send_events_pipeline=None,aws_s3=None):
    
        global kafka_queue, results_analytics_queue, results_events_queue,s3_client

        kafka_pipeline = None
        # aws_access_key=str(aws_s3["aws_access_key_id"])
        # aws_seceret_key=str(aws_s3["aws_access_key_id"])
        aws_region=str(aws_s3["region_name"])
        s3_bucket_name=str(aws_s3["BUCKET_NAME"])



        while True:
            try:
                # Try to create producer if it's None
                if kafka_pipeline is None:
                    logging.info("Attempting to connect to Kafka_Lets Try")
                    kafka_pipeline = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    
                    # Reliability settings
                    acks='all',                    # Wait for all replicas
                    retries=5,                     # Number of retries
                    retry_backoff_ms=1000,        # Retry backoff time
                    
                    # Performance tuning for 0.8MB messages
                    compression_type='gzip',       # Compress messages
                    batch_size=1048576,           # 1MB batch size
                    buffer_memory=67108864,       # 64MB buffer
                    max_request_size=1048576,     # 1MB max request
                    
                    # Timeouts
                    request_timeout_ms=30000,     # 30 seconds
                    max_block_ms=60000,          # 1 minute max blocking
                    
                    # Linger time (wait time for batching)
                    linger_ms=5000
                    )
                    time.sleep(20)
                    if kafka_pipeline is None:
                        logging.warning("Kafka connection failed, retrying in 60 seconds...")
                        time.sleep(60)
                        continue
                    else:
                        logging.info("Successfully connected to Kafka")
                # Process messages from all queues
                queues_and_topics = [
                    (kafka_queue, ai_engine_pipeline),
                    (results_analytics_queue, send_analytics_pipeline),
                    (results_events_queue, send_events_pipeline)
                ]

                while True:    
                    for queue_obj, topic in queues_and_topics:
                        try:
                            message = queue_obj.get_nowait()
                            if message is not None and topic != "None":
                                if str(topic) == str(send_events_pipeline):
                                    try:
                                        image_bytes = message["org_img"]
                                        labelled_image_bytes = make_labelled_image(message)
                                        image_s3_url, image_minio_url = upload_to_s3(s3_bucket_name,aws_region,image_bytes)
                                        print(image_s3_url, image_minio_url,"S3 and Minio URL")
                                        labelled_image_s3_url,labelled_image_minio_url = upload_to_s3(s3_bucket_name,aws_region,labelled_image_bytes)
                                        print(labelled_image_s3_url,labelled_image_minio_url,"Lablled S3 and Minio URL")
                                        # Send message with timeout
                                        # Extract category and subcategory from "Category-Subcategory"
                                        subcategory_full = message["absolute_bbox"][0]["subcategory"]
                                        parts = subcategory_full.split("-", 1)
                                        event_category = parts[0] if len(parts) > 0 else subcategory_full
                                        event_subcategory = parts[1] if len(parts) > 1 else ""
                                        message["org_img"] = image_s3_url
                                        #logging.info(f"{message},{image_s3_url}")
                                        try:
                                            if image_s3_url:
                                                kafka_pipeline.send(topic, message)
                                                kafka_pipeline.flush()  # Ensure message is actually sent
                                                print(message,"Sent Successfully")
                                        except:
                                            pass
                                            
                                        try:
                                            if labelled_image_minio_url:
                                                response = fill_form(
                                            'template.json',
                                            {
                                                "id": "7f148215-d1c5-4b83-bf58-3a2249d4b107",  
                                                "timestamp": message["datetimestamp_trackerid"],
                                                "image_original": f"{image_minio_url}",
                                                "image_labelled": f"{labelled_image_minio_url}", 
                                                "video_clip": "",       
                                                "remarks": "Violation Detected",
                                                "event_category": event_category,
                                                "whom_to_notify": "mohammed@arresto.in", 
                                                "camera_unique_id": "3acbb55c-d6cb-4470-8d9c-5d8de5223bea",
                                                "safety_captain_id": "varun@arresto.in",
                                                "incident_occured_on": message["datetimestamp_trackerid"],
                                                "incident_updated_on": message["datetimestamp_trackerid"],
                                                "action_status": "Pending",
                                                "event_subcategory": event_subcategory
                                            }
                                        )
                                                print(f"Successfully sent message to {topic},{response}")
                                            else:
                                                print(f"No Minio and Dashboard Online, Please check")
                                                
                                        except:
                                            print( "Its running but API server Needed")
                                            pass
                                        # Success - exit retry loop
                                    
                                    except (KafkaError, NoBrokersAvailable) as e:
                                        break
                                else:
                                    try:
                                        # Send message with timeout
                                        kafka_pipeline.send(topic, message)
                                        # kafka_pipeline.flush()  # Ensure message is actually sent
                                        #logging.info(f"{message},{topic}")
                                        #print(f"Successfully sent message to {topic}")
                                        # Success - exit retry loop
                                    
                                    except (KafkaError, NoBrokersAvailable) as e:
                                        break

                            
                        except queue.Empty:
                            continue  # No messages in this queue
                            
                        except Exception as e:
                            print(f"Error processing queue for {topic}: {str(e)}")
                            logging.error(f"Error processing queue for {topic}: {str(e)}")
                            # Return to outer loop to handle connection reset
                            break
                    time.sleep(0.5)
                time.sleep(60) 
                
            except (KafkaError, NoBrokersAvailable, socket.error) as e:
                time.sleep(60)
                logging.error(f"Unexpected error in Kafka pipeline: {str(e)}")
                if kafka_pipeline is not None:
                    try:
                        kafka_pipeline.close()
                    except:
                        pass
                    finally:
                        kafka_pipeline = None


def get_unique_tracker_ids(last_n_frames):
    """
    Return a set of unique tracker_ids across the last n frames.
    """
    unique_tracker_ids = set()
    for frame_tracker_ids in last_n_frames:
        unique_tracker_ids.update(frame_tracker_ids)
    return unique_tracker_ids

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Each box should be a list or tuple in the format:
    [xmin, ymin, xmax, ymax]
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the Intersection over Union by taking the intersection
    # area and dividing it by the sum of the prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou
# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data,frame_type):
    global last_n_frames_tracker_ids, kafka_queue
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        # print(frame_type,"Error")
        return Gst.PadProbeReturn.OK

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    try:
        if frame_type == "original":
            user_data.increment_original_frame_count()
            # print(user_data.original_frame_count)
            if format is not None and width is not None and height is not None:
                # Get video frame
                user_data.image = get_numpy_from_buffer(buffer, format, width, height)
                # user_data.image = cv2.cvtColor(user_data.image, cv2.COLOR_BGR2RGB)
                user_data.original_height=height
                user_data.original_width=width
                # print(height,width)
        
        elif frame_type == "pose_estimated":
            frame = None
            # Get the detections from the buffer
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

            # Get the keypoints
            keypoints = get_keypoints()

            if len(detections) != 0:
                
                xyxys=[]
                class_ids=[]
                class_names=[]
                confidences=[]
                pose_keypoints=[]

                #Scaling
                ratio=min(width/user_data.original_width,height/user_data.original_height)
                padx=int((width - user_data.original_width*ratio)/2)
                pady=int((height - user_data.original_height*ratio)/2)
                
                for detection in detections:
                   
                    label = detection.get_label()
                    bbox = detection.get_bbox()
                    confidence = detection.get_confidence()
                    pose_landmarks=detection.get_objects_typed(hailo.HAILO_LANDMARKS)
                    points = pose_landmarks[0].get_points()

                    x_min=min(max(int((bbox.xmin()*width-padx)/ratio),0),user_data.original_width)
                    y_min=min(max(int((bbox.ymin()*height-pady)/ratio),0),user_data.original_height)
                    x_max=min(max(int((bbox.xmax()*width-padx)/ratio),0),user_data.original_width)
                    y_max=min(max(int((bbox.ymax()*height-pady)/ratio),0),user_data.original_height)

                    scaled_keypoints = []
                    
                    if len(pose_landmarks) != 0:
                        # print("I am running Pose")
                        for pose_id in keypoints.keys():
                            keypoint_index = keypoints[pose_id]
                            kp=points[keypoint_index]
                            # Scale the x and y values using the same transformation as for the bbox.
                            kp_x = min(max(int((kp.x() * width - padx) / ratio), 0), user_data.original_width)
                            kp_y = min(max(int((kp.y() * height - pady) / ratio), 0), user_data.original_height)
                            conf=kp.confidence()
                            scaled_keypoints.append((kp_x, kp_y,conf))

                    #Appending the results from Detections
                    xyxys.append([x_min,y_min,x_max,y_max])
                    class_ids.append(detection.get_class_id())
                    class_names.append(label)
                    confidences.append(confidence)
                    pose_keypoints.append(scaled_keypoints)
                    
                result_to_tracker= result_to_tracker_OD(xyxys,confidences,class_ids,class_names)
                final_results = user_data.pose_tracker.update_with_detections(result_to_tracker)
                # print(final_results)
                #Updating in user data
                user_data.pose_tracker_ids=final_results.tracker_id
                user_data.pose_detection_boxes=xyxys
                user_data.pose_classes=class_names
                user_data.pose_keypoints=pose_keypoints


                # print(landmarks)
                # if len(landmarks) != 0:
                #     points = landmarks[0].get_points()
                #     for eye in ['left_eye', 'right_eye']:
                #         keypoint_index = keypoints[eye]
                #         point = points[keypoint_index]
                #         x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                #         y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                #         string_to_print = f"{eye}: x: {x:.2f} y: {y:.2f}\n"
                #         print(string_to_print)
                #         if user_data.use_frame:
                #             cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        elif frame_type=="processed":
            # Using the user_data to count the number of frames
            user_data.increment()
            string_to_print = f"Frame count: {user_data.get_count()}\n"


            # If the user_data.use_frame is set to True, we can get the video frame from the buffer
            frame = None
            if format is not None and width is not None and height is not None:
                # Get video frame
                user_data.model_image = get_numpy_from_buffer(buffer, format, width, height)

            # Get the detections from the buffer
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            user_data.time_stamp.append(time.time())
            # print(detections)
            # Parse the detections
            detection_count = 0
            xyxys=[]
            class_ids=[]
            class_names=[]
            confidences=[]

            #Scaling
            ratio=min(width/user_data.original_width,height/user_data.original_height)
            padx=int((width - user_data.original_width*ratio)/2)
            pady=int((height - user_data.original_height*ratio)/2)
            
            for detection in detections:
                label = detection.get_label()
                bbox = detection.get_bbox()
                confidence = detection.get_confidence()

                x_min=min(max(int((bbox.xmin()*width-padx)/ratio),0),user_data.original_width)
                y_min=min(max(int((bbox.ymin()*height-pady)/ratio),0),user_data.original_height)
                x_max=min(max(int((bbox.xmax()*width-padx)/ratio),0),user_data.original_width)
                y_max=min(max(int((bbox.ymax()*height-pady)/ratio),0),user_data.original_height)
                
                #Appending the results from Detections
                xyxys.append([x_min,y_min,x_max,y_max])
                class_ids.append(detection.get_class_id())
                class_names.append(label)
                confidences.append(confidence)
                # if label == "person":
                #     string_to_print += f"Detection: {label} {confidence:.2f}\n"
                detection_count += 1
            
            
            #Preparing results for trackers
            result_to_tracker= result_to_tracker_OD(xyxys,confidences,class_ids,class_names)
            final_results = user_data.tracker.update_with_detections(result_to_tracker)
            anchor_points = final_results.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            user_data.anchor_points_original=anchor_points
            
            # Updating the Activities Instance
            user_data.detection_boxes=final_results.xyxy
            user_data.classes=final_results.data["class_name"].tolist() if "class_name" in final_results.data else []
            user_data.tracker_ids=final_results.tracker_id
            
            if user_data.view_transformer is not None:
                user_data.anchor_points = user_data.view_transformer.transform_points(points=anchor_points).astype(int)
                #print(anchor_points, user_data.anchor_points)
           
            
            if xyxys:
                #Preparing results for sending to Server
                if int(user_data.asset_config["data_time_loop"]) != 0:
                    if (time.time() - user_data.current_time)>user_data.asset_config["data_time_loop"]:
                        img_byte=serialize_image(user_data.image)
                        final_json_results=json_format_serialization(final_results)
                        extra_data = {
                            "sensor_id":user_data.asset_config["sensor_id"],
                            "timestamp":datetime.now(user_data.ist_timezone).isoformat(),
                            "extra_data":{
                                "asset_tracking":final_json_results
                            }
                        }
                        message = create_message(img_byte,extra_data)
                        if kafka_queue.full():
                            lost_data=kafka_queue.get()
                            logging.info("Data getting Lost now")
                            kafka_queue.put(message)
                        else:
                            kafka_queue.put(message)
                        # print("Brother I am on Update")
                        user_data.current_time=time.time()


            # Last n frame tracker ids
            last_n_frames_tracker_ids.append(final_results.tracker_id.tolist())
            user_data.last_n_frame_tracker_ids=get_unique_tracker_ids(last_n_frames_tracker_ids)
            
            # Run each specified activity method using the same YOLO results
            current_time = time.time()
            for method in user_data.active_methods:
                method_name = method.__name__
                
                # Check if this method has timing control
                if method_name in user_data.method_intervals:
                    last_run = user_data.method_timing.get(method_name, 0)
                    interval = user_data.method_intervals[method_name]
                    
                    # Only run if enough time has passed
                    if current_time - last_run >= interval:
                        method()
                        user_data.method_timing[method_name] = current_time
                else:
                    # Run methods without timing control on every frame
                    method()

            # Saving Prev-pos
            for idx,(tracker_id,box) in enumerate(zip(final_results.tracker_id,final_results.xyxy)):
                user_data.prev_pos[tracker_id]=box
            
            # Try to reset the variables
            user_data.reset_and_cleaning_variables()
            # print(int((time.time()-user_data.cleaning_time_for_events)))
            if int((time.time()-user_data.cleaning_time_for_events)) > 120:
                #print("starting_cleaning")
                user_data.cleaning_events_data_with_last_frames() 
                user_data.cleaning_time_for_events=time.time()

            # labels = [f"#{tracker_id} {class_name_f}" for class_name_f, tracker_id in zip(final_results.data['class_name'], final_results.tracker_id)]
            # frame = user_data.box_annotator.annotate(user_data.image, detections=final_results)
            # frame = user_data.label_annotator.annotate(user_data.image, detections=final_results, labels=labels)
            # frame = user_data.trace_annotator.annotate(user_data.image, detections=final_results)
            
            if detection_count==0:
                user_data.reset_tracker+=1
                if user_data.reset_tracker>user_data.reset_threshold:
                    user_data.tracker.reset()
            else:
                user_data.reset_tracker=0

            # if user_data.use_frame:
                # Note: using imshow will not work here, as the callback function is not running in the main thread
                # Let's print the detection count to the frame
                # cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # # Example of how to use the new_variable and new_function from the user_data
                # # Let's print the new_variable and the result of the new_function to the frame
                # # cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # # Convert the frame to BGR
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # user_data.set_frame(user_data.image)
            # print(string_to_print)
    except Exception as e:
        logging.info(f"An error occurred: {e}")
    return Gst.PadProbeReturn.OK

# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
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

    return keypoints

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

if __name__ == "__main__":

    # Logging and put system sleep at reboot
    setup_logging()

    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    
     # Load the JSON data from the file
    with open('configuration.json', 'r') as file:
        config = json.load(file)

    #Getting Kafka Configuration
    kafka_variables= config.get("kafka_variables", {})
    bootstrap_servers=str(kafka_variables["bootstrap_servers"])
    websocket_url=str(config.get("websocket_url"))
    ai_engine_pipeline = str(kafka_variables.get("ai_engine_pipeline"))
    send_analytics_pipeline=str(kafka_variables.get("send_analytics_pipeline"))
    send_events_pipeline=str(kafka_variables.get("send_events_pipeline"))
    aws_s3=kafka_variables.get("AWS_S3")
    

    #Creating Thread for Kafka
    kafka_thread=Thread(target=send_data_to_kafka_broker,kwargs={'websocket_url':websocket_url,'bootstrap_servers':bootstrap_servers,'ai_engine_pipeline':ai_engine_pipeline,'send_analytics_pipeline':send_analytics_pipeline,'send_events_pipeline':send_events_pipeline,'aws_s3':aws_s3,})
    kafka_thread.daemon = True
    kafka_thread.start()
    #Getting Data available
    sensor_id=config.get("sensor_id")
    available_activities = config.get("available_activities")
    active_activities = config.get("active_activities")
    # print(active_activities)
    confidence=config["model_details"]["confidence"]
    user_data.sensor_id = sensor_id
    user_data.results_analytics_queue = results_analytics_queue
    user_data.results_events_queue = results_events_queue
    user_data.reset_threshold=kafka_variables["reset_threshold"]

    # Extract zones data for each activity in activities_data
    zones_data = {}
    parameters_data={}
    violation_id_data={}
    for activity, details in config["activities_data"].items():
        if "zones" in details and activity in active_activities:
            if activity == "entry_exit_WLE_logs":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: [Polygon(coords[0]),Polygon(coords[1])] for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.entry_exit_WLE_logs_data[zone_name]={}
                    user_data.entry_exit_WLE_logs_data[zone_name]["entry_ids"]={}
                    user_data.entry_exit_WLE_logs_data[zone_name]["exit_ids"]={}
                    user_data.entry_exit_WLE_logs_data[zone_name]["entry_count"]={}
                    user_data.entry_exit_WLE_logs_data[zone_name]["exit_count"]={}
                    for object_class in details["parameters"]["subcategory_mapping"]:
                        user_data.entry_exit_WLE_logs_data[zone_name]["entry_ids"][object_class]=[]
                        user_data.entry_exit_WLE_logs_data[zone_name]["exit_ids"][object_class]=[]
                        user_data.entry_exit_WLE_logs_data[zone_name]["entry_count"][object_class]=0
                        user_data.entry_exit_WLE_logs_data[zone_name]["exit_count"][object_class]=0
                parameters_data[activity]=details["parameters"]
                        

            elif activity == "workplace_area_occupancy":
                parameters_data[activity]=details["parameters"]
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.workplace_area_occupancy_data[zone_name]={}
                    user_data.workplace_area_occupancy_data[zone_name]["capacity"]=details["capacity"][zone_name]
                    user_data.workplace_area_occupancy_data[zone_name]["occupancy"]=0
                    
            elif activity == "people_gathering":
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.people_gathering_data[zone_name]={}
                    user_data.people_gathering_data[zone_name]["frame_accuracy"]=0
                    user_data.people_gathering_data[zone_name]["last_time"]=time.time()
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]
                user_data.active_activities_for_cleaning.append(activity)
                
            
            elif activity=="workforce_efficiency":
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                parameters_data[activity]=details["parameters"]
                for zone_name in details["zones"].keys():
                    user_data.workforce_efficiency_data[zone_name]=0
                user_data.workforce_efficiency_data["current_frame_count"]=0
                user_data.workforce_efficiency_data["send_data_frame_limit"]=kafka_variables["kafka_interval_for_analytics"]
                user_data.workforce_efficiency_data["timestamp"]=datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
                user_data.workforce_efficiency_data["time"]=time.time()

            elif activity=="desk_occupancy":
                parameters_data[activity]=details["parameters"]
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.desk_occupancy_data[zone_name]=0

            elif activity=="resource_utilization":
                parameters_data[activity]=details["parameters"]
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.resource_utilization_data[zone_name]=0
            
            elif activity=="traffic_overspeeding":
                SOURCE = np.array(details["parameters"]["SOURCE"])
                TARGET_WIDTH = details["parameters"]["TARGET_WIDTH"]
                TARGET_HEIGHT = details["parameters"]["TARGET_HEIGHT"]
                TARGET = np.array([[0, 0],[TARGET_WIDTH - 1, 0],[TARGET_WIDTH - 1, TARGET_HEIGHT - 1],[0, TARGET_HEIGHT - 1]])
                user_data.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]
                user_data.active_activities_for_cleaning.append(activity)
                user_data.time_stamp=deque(maxlen=details["parameters"]["frame_to_track"])
                user_data.traffic_overspeeding_data=defaultdict(lambda: deque(maxlen=details["parameters"]["frame_to_track"]))
                user_data.traffic_overspeeding_speed_data=defaultdict(SortedList)
                user_data.anprscan_line_area=Polygon(details["parameters"]["anprarea"])
            
            elif activity=="traffic_overspeeding_distancewise":
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]
                user_data.active_activities_for_cleaning.append(activity)
                user_data.traffic_overspeeding_distancewise_data=defaultdict(lambda: defaultdict(dict))
                user_data.distancewise_tracking=[]
                parameters_data["traffic_overspeeding_distancewise"]["lines_length"]={}
                user_data.time_stamp=deque(maxlen=20)
                for lane_name,coordinates in parameters_data[activity]["lines"].items():
                    total_distance = 0
                    # Iterate over the pairs of coordinates in the line
                    for i in range(len(coordinates) - 1):
                        x1, y1 = coordinates[i]
                        x2, y2 = coordinates[i + 1]
                        total_distance += user_data.calculate_distance(x1, y1, x2, y2)  # Calculate distance for each segment
                    
                    parameters_data[activity]["lines_length"][lane_name] = total_distance
            
            elif activity=="ppe":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.ppe_data[zone_name]={}
                violation_id_data[activity]={}
                for acts in details["parameters"]["subcategory_mapping"]:
                    violation_id_data[activity][acts]=[]
                parameters_data[activity]=details["parameters"]
                user_data.active_activities_for_cleaning.append(activity)
            
            elif activity=="perimeter_monitoring":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.perimeter_monitoring_data[zone_name]={}
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]
                user_data.active_activities_for_cleaning.append(activity)

            elif activity=="climbing":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.climbing_data[zone_name]={}
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]

            
            elif activity=="time_based_unauthorized_access":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.climbing_data[zone_name]={}
                parameters_data[activity]=details["parameters"]
                parameters_data[activity]["last_check_time"]=time.time()
                violation_id_data[activity]=[]
        
            elif activity=="vehicle_interaction":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.vehicle_interaction_data[zone_name]={}
                parameters_data[activity]=details["parameters"]
                violation_id_data[activity]=[]
                user_data.active_activities_for_cleaning.append(activity)
        
            elif activity=="person_violations":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.person_violations_data[zone_name]={}
                   
                violation_id_data[activity]={}
                for acts in details["parameters"]["subcategory_mapping"]:
                    violation_id_data[activity][acts]=[]
                parameters_data[activity]=details["parameters"]
                user_data.active_activities_for_cleaning.append(activity)

            elif activity=="fire_and_smoke":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.fire_and_smoke_data[zone_name]={}
                    user_data.fire_and_smoke_data[zone_name]["frame_count"]=0
                    user_data.fire_and_smoke_data[zone_name]["frame_count_last_time"]=time.time()
                    user_data.fire_and_smoke_data[zone_name]["alert_interval"]=time.time()
                parameters_data[activity]=details["parameters"]


            elif activity == "unattended_area":
                # Convert zone lists of (x, y) tuples to shapely Polygon objects
                zones_data[activity] = {zone: Polygon(coords) for zone, coords in details["zones"].items()}
                for zone_name in details["zones"].keys():
                    user_data.unattended_area_data[zone_name] = {}
                    user_data.unattended_area_data[zone_name]["last_person_time"]=time.time()
                    user_data.unattended_area_data[zone_name]["person_present"] = False
                parameters_data[activity] = details["parameters"]
                violation_id_data[activity] = []
                user_data.active_activities_for_cleaning.append(activity)
            
            elif activity == "camera_tampering":
                parameters_data[activity] = details["parameters"]
                user_data.camera_tampering_data["alert_interval"]=time.time()
                user_data.camera_tampering_data["frame_count"]=0

    print(user_data.active_activities_for_cleaning,active_activities )
    #Assigning Zones Data to Activity Instance
    user_data.zone_data=zones_data
    user_data.parameters_data = parameters_data
    user_data.violation_id_data = violation_id_data

    # Making Active Methods 
    active_methods=[]
    for activity in active_activities:
        # Retrieve the method from the Activities instance if it exists
        activity_func = getattr(user_data, activity, None)
        if callable(activity_func):
            active_methods.append(activity_func)
        else:
            logging.info(f"Warning: Activity '{activity}' not recognized in Activities class.")

    user_data.active_methods=active_methods

    app = GStreamerDetectionApp(app_callback, user_data)
    try:
        app.run()
    finally:
        kafka_thread.join()
