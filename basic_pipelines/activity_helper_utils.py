import os
import cv2
import numpy as np
import base64
from collections import deque
from shapely.geometry import Point, Polygon
from typing import List, Tuple, Dict, Any, Optional
import time
from datetime import datetime
import pytz


def is_bottom_in_zone(anchor_point: Tuple[float, float], zone_polygon: Polygon) -> bool:
    """
    Check if the center of the vehicle's bounding box is inside a zone.
    
    Args:
        anchor_point: Center point of vehicle (x, y)
        zone_polygon: Shapely Polygon object for the zone
        
    Returns:
        True if vehicle is in zone, False otherwise
    """
    xb_center, y_center = anchor_point
    bottom_point = Point(xb_center, y_center)
    return zone_polygon.contains(bottom_point)

def is_object_in_zone(object_coordinates, zone_polygon):
        """
        Check if the center of the person's bounding box is inside a zone.
        person_coordinates: [xmin, ymin, xmax, ymax]
        zone_polygon: shapely Polygon object for the zone
        """
        x_center = (object_coordinates[0] + object_coordinates[2]) / 2
        y_center = (object_coordinates[1] + object_coordinates[3]) / 2
        object_center = Point(x_center, y_center)
        return zone_polygon.contains(object_center)

def xywh_original_percentage(box: List[float], original_width: int, original_height: int) -> List[float]:
    """
    Convert bounding box coordinates to percentage of original image dimensions.
    
    Args:
        box: Bounding box [xmin, ymin, xmax, ymax]
        original_width: Original image width
        original_height: Original image height
        
    Returns:
        Bounding box as percentages [x%, y%, width%, height%]
    """
    min_x, min_y, max_x, max_y = box[0], box[1], box[2], box[3]
    xywh = [
        float(min_x * 100 / original_width),
        float(min_y * 100 / original_height),
        float((max_x - min_x) * 100 / original_width),
        float((max_y - min_y) * 100 / original_height)
    ]
    return xywh
    
# calculate iou
def calculate_iou(self, boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA: First bounding box in format [x_min, y_min, x_max, y_max]
        boxB: Second bounding box in format [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU value between 0 and 1
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
    
    # Compute the Intersection over Union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def merge_box(self,box1,box2):
        # Compute the union of the two bounding boxes
        xmin = min(box1[0], box2[0])
        ymin = min(box1[1], box2[1])
        xmax = max(box1[2], box2[2])
        ymax = max(box1[3], box2[3])
        union_box = [xmin, ymin, xmax, ymax]
        return union_box
    
# bottom center
def bottom_center(box):
    return [int((box[0] + box[2]) / 2), int(box[3])]

# relay
def init_relay(parent, parameters):
    """
    Initialize relay safely.
    Returns (relay_handler, switch_relay_list)
    """
    if parameters["relay"] == 1:
        try:
            if parent.relay_handler.device is None:
                success = parent.relay_handler.initiate_relay()
                if not success:
                    print("⚠️ Relay device not available. Continuing without relay control.")
                    return None, []
            return parent.relay_handler, parameters["switch_relay"]

        except Exception as e:
            print(f"⚠️ Relay initialization failed: {e}")
            return None, []

    return None, []

def trigger_relay(relay, switch_relay):
    """
    Turn ON relay channels safely and update start_time.
    """   
    if relay is None:
        return

    try:
        status = relay.state(0)
        true_indexes = [
            (i + 1) for i, x in enumerate(status)
            if isinstance(x, bool) and x is True
        ]

        for index in switch_relay:
            if index not in true_indexes:
                relay.state(index, on=True)
            relay.start_time[index] = time.time()

    except Exception as e:
        print(f"⚠️ Relay operation failed: {e}.  Continuing without relay control.")
        
def relay_auto_off(relay, switch_relay):
    """
    Call auto-off logic if relay exists.
    """
    if relay is None:
        return

    try:
        relay.check_auto_off(switch_relay)
    except Exception as e:
        print(f"⚠️ Relay auto-off failed: {e}.  Continuing without relay control.")



# activity active time

def activity_active_time(parameters, timezone):
    """
    Determines whether an activity should run based on scheduled_time config.

    parameters: activity parameters dict from configuration.json
    timezone: pytz timezone object already created in activity

    scheduled_time format (optional):
    [
        {
            "time_start_end": [["HH:MM", "HH:MM"], ...],
            "days": ["Monday", "Tuesday", ...]
        }
    ]
    """

    scheduled_time = parameters.get("scheduled_time")

    # No scheduling config → always active
    if not scheduled_time:
        return True

    now = datetime.now(timezone)
    current_day = now.strftime("%A")
    current_time = now.time()

    for schedule in scheduled_time:
        days = schedule.get("days")
        time_ranges = schedule.get("time_start_end")

        # Day check
        if days:
            if current_day not in days:
                continue  # this schedule does not match
        # else: no days → all days allowed

        # Time check
        if time_ranges:
            for start_str, end_str in time_ranges:
                start_time = datetime.strptime(start_str, "%H:%M").time()
                end_time = datetime.strptime(end_str, "%H:%M").time()

                if start_time <= current_time <= end_time:
                    return True
        else:
            # No time restriction → whole day allowed
            return True

    # No schedule matched
    return False
    
######################### helper fucntion for loitering #############
def extract_median_color(image, bbox):
    """Extract median RGB color from ROI defined by bbox in the frame."""
    if image is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    median_rgb = np.median(roi_rgb.reshape(-1, 3), axis=0)
    return median_rgb


def uniform_validation(rgb_value, category, hsv_target, hsv_tolerance):
    """Check if the given RGB color matches the HSV target for a category."""
    if rgb_value is None:
        return False
    bgr = np.uint8([[rgb_value[::-1]]])
    h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    target    = hsv_target.get(category, [0, 0, 0])
    tolerance = hsv_tolerance.get(category, [0, 0, 0])
    hue_diff  = min(abs(int(h) - target[0]), 180 - abs(int(h) - target[0]))
    sat_diff  = abs(int(s) - target[1])
    val_diff  = abs(int(v) - target[2])
    return hue_diff <= tolerance[0] and sat_diff <= tolerance[1] and val_diff <= tolerance[2]
    
###################################################################

############# for running detection ##################################
def get_interpolated_ppm(calibration_data, y_coord):
    below = [p for p in calibration_data if p["y"] <= y_coord]
    above = [p for p in calibration_data if p["y"] > y_coord]
    if not below:
        return above[0]["ppm"]
    if not above:
        return below[-1]["ppm"]
    p1 = below[-1]
    p2 = above[0]
    if p1["y"] == p2["y"]:
        return p1["ppm"]
    return p1["ppm"] + (p2["ppm"] - p1["ppm"]) * ((y_coord - p1["y"]) / (p2["y"] - p1["y"]))
    
    
################## for light detection #################################
def get_zone_grayscale(image, polygon):
    """Extract masked grayscale of a zone polygon from image."""
    if image is None:
        return None
    # polygon is already a Shapely Polygon object
    mask   = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(
        [[(int(x), int(y)) for x, y in polygon.exterior.coords]],
        dtype=np.int32
    )
    cv2.fillPoly(mask, points, 255)
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zone_gray = cv2.bitwise_and(gray, gray, mask=mask)
    return zone_gray


def compute_histogram(zone_gray):
    """Compute normalized histogram from a grayscale zone image."""
    hist = cv2.calcHist([zone_gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
