import os
import cv2
import numpy as np
import base64
from collections import deque
from shapely.geometry import Point, Polygon
from typing import List, Tuple, Dict, Any, Optional

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