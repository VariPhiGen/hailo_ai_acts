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

def is_activity_active(parameters, timezone):
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

