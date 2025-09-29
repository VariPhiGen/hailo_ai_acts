import os
import cv2
import numpy as np
import base64
from collections import deque
from shapely.geometry import Point, Polygon
from typing import List, Tuple, Dict, Any, Optional


def setup_logging():
    """No-op function - logging is completely disabled."""
    pass


def serialize_image(image: np.ndarray) -> str:
    """
    Serialize image to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string 
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, buffer = cv2.imencode('.jpeg', image, encode_param)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def encode_frame_to_base64(image: np.ndarray) -> str:
    """
    Encode frame to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string
    """
    _, buffer = cv2.imencode('.jpg', image)
    jpeg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpeg_as_text


def encode_frame_to_bytes(image: np.ndarray, quality: int = 95) -> bytes:
    """
    Encode frame to bytes.
    
    Args:
        image: Image as numpy array
        quality: JPEG quality (1-100)
        anpr: Optional ANPR image
        
    Returns:
        Image as bytes
    """
    if quality < 100:
        # Convert from BGR to RGB
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
    else:
        _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()


def is_vehicle_in_zone(anchor_point: Tuple[float, float], zone_polygon: Polygon) -> bool:
    """
    Check if the center of the vehicle's bounding box is inside a zone.
    
    Args:
        anchor_point: Center point of vehicle (x, y)
        zone_polygon: Shapely Polygon object for the zone
        
    Returns:
        True if vehicle is in zone, False otherwise
    """
    x_center, y_center = anchor_point
    vehicle_point = Point(x_center, y_center)
    return zone_polygon.contains(vehicle_point)


def crop_image_numpy(image_array: np.ndarray, bounding_box: Tuple[float, float, float, float], pad: int = 50) -> np.ndarray:
    """
    Crop image safely with bounds checking and padding.
    
    Args:
        image_array: Image as numpy array (H, W, C)
        bounding_box: (left, upper, right, lower)
        pad: padding in pixels to extend the crop on all sides
    
    Returns:
        Cropped image as numpy array. Falls back to unclipped box or full image if invalid.
    """
    h, w = image_array.shape[:2]
    left, upper, right, lower = bounding_box

    # Normalize coordinates (ensure left<=right, upper<=lower)
    x_min = min(left, right)
    x_max = max(left, right)
    y_min = min(upper, lower)
    y_max = max(upper, lower)

    # First try padded crop with clamping
    x1 = max(2, int(x_min))
    y1 = max(2, int(y_min))
    x2 = min(w-1, int(x_max + pad))
    y2 = min(h-1, int(y_max + pad))

    if x2 > x1 and y2 > y1:
        cropped = image_array[y1:y2, x1:x2]
        if cropped.size > 0:
            return cropped

    # Fallback to unclipped, unpadded box within bounds
    x1 = max(0, int(x_min))
    y1 = max(0, int(y_min))
    x2 = min(w, int(x_max+20))
    y2 = min(h, int(y_max+20))

    if x2 > x1 and y2 > y1:
        cropped = image_array[y1:y2, x1:x2]
        if cropped.size > 0:
            return cropped

    # Last resort: return the full frame to avoid empty crops
    return image_array


def find_clusters_by_zone(gathered_tracker_ids: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[List[int]]]:
    """
    Find clusters of tracker IDs by zone using Union-Find algorithm.
    
    Args:
        gathered_tracker_ids: Dictionary mapping zone names to pairs of tracker IDs
        
    Returns:
        Dictionary mapping zone names to lists of tracker ID clusters
    """
    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

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


def cleaning_list_with_last_frames(list_of_ids: List[int], last_n_frame_tracker_ids: deque) -> List[int]:
    """
    Clean list by keeping only IDs that are in the last n frames.
    
    Args:
        list_of_ids: List of tracker IDs to clean
        last_n_frame_tracker_ids: Deque containing tracker IDs from last n frames
        
    Returns:
        Filtered list of tracker IDs
    """
    results = []
    for tracker_id in list_of_ids:
        if tracker_id in last_n_frame_tracker_ids:
            results.append(tracker_id)
    return results


def get_unique_tracker_ids(last_n_frames: List[List[int]]) -> set:
    """
    Return a set of unique tracker_ids across the last n frames.
    
    Args:
        last_n_frames: List of lists containing tracker IDs from each frame
        
    Returns:
        Set of unique tracker IDs
    """
    unique_tracker_ids = set()
    for frame_tracker_ids in last_n_frames:
        unique_tracker_ids.update(frame_tracker_ids)
    return unique_tracker_ids


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA: First bounding box [xmin, ymin, xmax, ymax]
        boxB: Second bounding box [xmin, ymin, xmax, ymax]
        
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


def calculate_euclidean_distance(coord_start: Tuple[float, float], coord_end: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two coordinates.
    
    Args:
        coord_start: Starting coordinates (x, y)
        coord_end: Ending coordinates (x, y)
        
    Returns:
        Euclidean distance
    """
    coord_start = np.array(coord_start)
    coord_end = np.array(coord_end)
    distance = np.linalg.norm(coord_end - coord_start)
    return distance


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the distance between two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Distance between points
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def project_point_on_line(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> Tuple[float, float]:
    """
    Project a point onto a line segment and return the projection point.
    
    Args:
        x1, y1: First point of line segment
        x2, y2: Second point of line segment
        px, py: Point to project
        
    Returns:
        Projected point coordinates (x, y)
    """
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


def closest_line_projected_distance(vehicle_path: List[Tuple[float, float]], lines: Dict[str, List[List[float]]]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Calculate the closest line's projected distance from the midpoint of the vehicle's path.
    
    Args:
        vehicle_path: List of two points representing vehicle path [(x1, y1), (x2, y2)]
        lines: Dictionary mapping lane names to line coordinates
        
    Returns:
        Tuple of (travelled_distance, total_distance_travelled, closest_lane)
    """
    (x1, y1), (x2, y2) = vehicle_path  # Vehicle path coordinates
    
    # Calculate the midpoint of the vehicle's path
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    
    min_distance = float('inf')
    close_line = None
    total_distance_travelled = None
    travelled_distance = None
    closest_lane = None
    
    # Loop through each line to calculate the projected distance from the midpoint of the vehicle path
    for lane_name, line in lines.items():
        [[x1_line, y1_line], [x2_line, y2_line]] = line  # Line coordinates
        
        # Project the midpoint of the vehicle path onto the line
        proj_mid_x, proj_mid_y = project_point_on_line(x1_line, y1_line, x2_line, y2_line, midpoint_x, midpoint_y)
        
        # Calculate the distance between the midpoint and the projected point on the line
        distance = calculate_distance(midpoint_x, midpoint_y, proj_mid_x, proj_mid_y)
        
        # Track the minimum distance (closest line's distance)
        if distance < min_distance:
            min_distance = distance
            close_line = line
            closest_lane = lane_name
    
    if close_line is not None:
        [[x1_line, y1_line], [x2_line, y2_line]] = close_line
        proj_x1, proj_y1 = project_point_on_line(x1_line, y1_line, x2_line, y2_line, x1, y1)
        proj_x2, proj_y2 = project_point_on_line(x1_line, y1_line, x2_line, y2_line, x2, y2)
        travelled_distance = calculate_distance(proj_x1, proj_y1, proj_x2, proj_y2)
        total_distance_travelled = calculate_distance(x1, y1, x2, y2)
    
    return travelled_distance, total_distance_travelled, closest_lane


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


def om_centerbott(box: List[float]) -> Tuple[int, int]:
    """
    Given a bounding box [xmin, ymin, xmax, ymax],
    return the bottom-center point (x, y) as integers.
    
    Args:
        box: Bounding box [xmin, ymin, xmax, ymax]
        
    Returns:
        Bottom-center point (x, y)
    """
    x_center = int((box[0] + box[2]) / 2)
    y_bottom = int(box[3])
    return (x_center, y_bottom)


def json_format_serialization(final_results) -> Dict[str, Any]:
    """
    Serialize detection results to JSON format.
    
    Args:
        final_results: Detection results object
        
    Returns:
        Dictionary with serialized results
    """
    final_json_results = {}
    final_json_results["xyxy"] = final_results.xyxy.astype(int).tolist()
    final_json_results["confidence"] = final_results.confidence.tolist()
    final_json_results["class_id"] = final_results.class_id.tolist()
    final_json_results["tracker_id"] = final_results.tracker_id.tolist()
    final_json_results["class_name"] = final_results.data["class_name"].tolist() if "class_name" in final_results.data else []
    return final_json_results 
