import os
import cv2
import numpy as np
import base64
from collections import deque
from shapely.geometry import Point, Polygon
from typing import List, Tuple, Dict, Any, Optional


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