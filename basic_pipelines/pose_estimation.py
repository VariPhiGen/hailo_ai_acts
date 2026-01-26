# from pathlib import Path
# import gi
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GLib
# import os
# import numpy as np
# import cv2
# import hailo

# from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
# from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
# from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

# # -----------------------------------------------------------------------------------------------
# # User-defined class to be used in the callback function
# # -----------------------------------------------------------------------------------------------
# # Inheritance from the app_callback_class
# class user_app_callback_class(app_callback_class):
#     def __init__(self):
#         super().__init__()

# # -----------------------------------------------------------------------------------------------
# # User-defined callback function
# # -----------------------------------------------------------------------------------------------

# # This is the callback function that will be called when data is available from the pipeline
# def app_callback(pad, info, user_data):
#     # Get the GstBuffer from the probe info
#     buffer = info.get_buffer()
#     # Check if the buffer is valid
#     if buffer is None:
#         return Gst.PadProbeReturn.OK

#     # Using the user_data to count the number of frames
#     user_data.increment()
#     string_to_print = f"Frame count: {user_data.get_count()}\n"

#     # Get the caps from the pad
#     format, width, height = get_caps_from_pad(pad)

#     # If the user_data.use_frame is set to True, we can get the video frame from the buffer
#     frame = None
#     if user_data.use_frame and format is not None and width is not None and height is not None:
#         # Get video frame
#         frame = get_numpy_from_buffer(buffer, format, width, height)

#     # Get the detections from the buffer
#     roi = hailo.get_roi_from_buffer(buffer)
#     detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

#     # Get the keypoints
#     keypoints = get_keypoints()

#     # Parse the detections
#     for detection in detections:
#         label = detection.get_label()
#         bbox = detection.get_bbox()
#         confidence = detection.get_confidence()
#         if label == "person":
#             # Get track ID
#             track_id = 0
#             track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
#             if len(track) == 1:
#                 track_id = track[0].get_id()
#             string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")

#             # Pose estimation landmarks from detection (if available)
#             landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
#             if len(landmarks) != 0:
#                 points = landmarks[0].get_points()
#                 for eye in ['left_eye', 'right_eye']:
#                     keypoint_index = keypoints[eye]
#                     point = points[keypoint_index]
#                     x = int((point.x() * bbox.width() + bbox.xmin()) * width)
#                     y = int((point.y() * bbox.height() + bbox.ymin()) * height)
#                     string_to_print += f"{eye}: x: {x:.2f} y: {y:.2f}\n"
#                     if user_data.use_frame:
#                         cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

#     if user_data.use_frame:
#         # Convert the frame to BGR
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         user_data.set_frame(frame)

#     print(string_to_print)
#     return Gst.PadProbeReturn.OK

# # This function can be used to get the COCO keypoints coorespondence map
# def get_keypoints():
#     """Get the COCO keypoints and their left/right flip coorespondence map."""
#     keypoints = {
#         'nose': 0,
#         'left_eye': 1,
#         'right_eye': 2,
#         'left_ear': 3,
#         'right_ear': 4,
#         'left_shoulder': 5,
#         'right_shoulder': 6,
#         'left_elbow': 7,
#         'right_elbow': 8,
#         'left_wrist': 9,
#         'right_wrist': 10,
#         'left_hip': 11,
#         'right_hip': 12,
#         'left_knee': 13,
#         'right_knee': 14,
#         'left_ankle': 15,
#         'right_ankle': 16,
#     }

#     return keypoints

# if __name__ == "__main__":
#     project_root = Path(__file__).resolve().parent.parent
#     env_file     = project_root / ".env"
#     env_path_str = str(env_file)
#     os.environ["HAILO_ENV_FILE"] = env_path_str
#     # Create an instance of the user app callback class
#     user_data = user_app_callback_class()
#     app = GStreamerPoseEstimationApp(app_callback, user_data)
#     app.run()


########################################################################################

# import cv2
# import numpy as np
# from hailo_platform import (
#     HEF, VDevice, ConfigureParams,
#     InputVStreamParams, OutputVStreamParams,
#     InferVStreams, FormatType, HailoStreamInterface
# )

# # --- based on coco keypoints ---
# COCO_PARTS = [
#     "nose","left_eye","right_eye","left_ear","right_ear",
#     "left_shoulder","right_shoulder",
#     "left_elbow","right_elbow",
#     "left_wrist","right_wrist",
#     "left_hip","right_hip",
#     "left_knee","right_knee",
#     "left_ankle","right_ankle"
# ]

# POSE_SKELETON = [
#     (5,7),(7,9),(6,8),(8,10),
#     (5,6),(5,11),(6,12),
#     (11,13),(13,15),(12,14),(14,16)
# ]

# # --- pose estimator ---
# class PoseEstimator:
#     def __init__(self, hef_path):
#         self.hef = HEF(hef_path)
#         self.device = VDevice()

#         self.network_group = self.device.configure(
#             self.hef,
#             ConfigureParams.create_from_hef(
#                 hef=self.hef,
#                 interface=HailoStreamInterface.PCIe
#             )
#         )[0]

#         self.input_params = InputVStreamParams.make(
#             self.network_group, format_type=FormatType.UINT8
#         )
#         self.output_params = OutputVStreamParams.make(
#             self.network_group, format_type=FormatType.FLOAT32
#         )

#         info = self.hef.get_input_vstream_infos()[0]
#         self.in_w = info.shape[1]
#         self.in_h = info.shape[0]

#     def infer_batch(self, crops):
#         resized = [cv2.resize(img, (self.in_w, self.in_h)) for img in crops]
#         batch = np.stack(resized, axis=0)

#         # ✅ CORRECT usage for your HailoRT
#         with InferVStreams(
#             self.network_group,
#             self.input_params,
#             self.output_params
#         ) as infer:
#             return infer.infer(batch)


# # --- postprocess ---
# def parse_pose_outputs(raw):
#     poses = []
#     for person in raw[0]:
#         kp = []
#         for i, name in enumerate(COCO_PARTS):
#             x, y, c = person[i]
#             if c < 0.3:
#                 continue
#             kp.append({"name": name, "x": int(x), "y": int(y), "conf": float(c)})
#         poses.append(kp)
#     return poses

# # --- async runner ---
# def run_pose_async(user_data, crops, meta):
#     outputs = user_data.pose_estimator.infer_batch(crops)

#     parsed = parse_pose_outputs(outputs)  # see next step

#     for (tid, x1, y1, x2, y2), keypoints in zip(meta, parsed):
#         # map back to full image
#         for k in keypoints:
#             k["x"] += x1
#             k["y"] += y1

#         user_data.latest_pose[tid] = keypoints



"""
Pose Estimation Module - Integrated into Detection Pipeline
Uses Hailo filter + secondary hailonet for pose inference
"""

import hailo
import cv2
import numpy as np

POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def create_pose_inference_branch(pipeline, detection_output, user_data):
    """
    Adds pose estimation branch to existing detection pipeline
    
    Args:
        pipeline: Main GStreamer pipeline
        detection_output: Pad from detection hailonet output
        user_data: Shared data object
    
    Returns:
        pose_sink: Sink element for pose results
    """
    from gi.repository import Gst
    
    # Create elements for pose branch
    tee = Gst.ElementFactory.make("tee", "pose_tee")
    queue_pose = Gst.ElementFactory.make("queue", "queue_pose")
    hailofilter = Gst.ElementFactory.make("hailofilter", "person_filter")
    pose_net = Gst.ElementFactory.make("hailonet", "pose_net")
    pose_sink = Gst.ElementFactory.make("fakesink", "pose_sink")
    
    # Add to pipeline
    for elem in [tee, queue_pose, hailofilter, pose_net, pose_sink]:
        pipeline.add(elem)
    
    # Configure hailofilter to extract person crops
    hailofilter.set_property('function-name', 'filter_person_class')
    hailofilter.set_property('config-string', 'person')
    
    # Configure pose hailonet
    pose_net.set_property('hef-path', user_data.pose_hef_path)
    pose_net.set_property('batch-size', 1)
    pose_net.set_property('nms-iou-threshold', 0.45)
    
    # Link: detection → tee → filter → pose_net → sink
    detection_output.link(tee)
    tee.link(queue_pose)
    queue_pose.link(hailofilter)
    hailofilter.link(pose_net)
    pose_net.link(pose_sink)
    
    return pose_sink

def setup_pose_probe(sink_element, user_data):
    """
    Attaches probe callback to pose sink
    
    Args:
        sink_element: Pose pipeline sink
        user_data: Shared data object
    """
    from gi.repository import Gst
    
    sink_pad = sink_element.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, pose_results_callback, user_data)

def pose_results_callback(pad, info, user_data):
    """
    Processes pose keypoints from Hailo metadata
    Stores results in user_data.pose_results
    """
    from gi.repository import Gst
    
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        landmarks = roi.get_objects_typed(hailo.HAILO_LANDMARKS)
        
        for landmark in landmarks:
            # Get tracker ID from parent detection
            parent_detection = landmark.get_parent()
            if parent_detection:
                track = parent_detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                if track:
                    tracker_id = track[0].get_id()
                    
                    # Extract keypoints
                    points = landmark.get_points()
                    keypoints = [{'x': int(p.x()), 'y': int(p.y()), 'confidence': p.confidence()} 
                                for p in points]
                    
                    # Store in shared dict
                    user_data.pose_results[tracker_id] = keypoints
    
    except Exception as e:
        print(f"⚠️ Pose processing error: {e}")
    
    return Gst.PadProbeReturn.OK

def draw_pose_on_frame(image, pose_results):
    """
    Draws pose skeleton on frame
    Call this from detection pipeline's frame callback
    
    Args:
        image: numpy array (frame)
        pose_results: dict {tracker_id: keypoints}
    
    Returns:
        Modified image with pose overlay
    """
    for tracker_id, keypoints in pose_results.items():
        # Draw skeleton connections
        for (a, b) in POSE_SKELETON:
            if a < len(keypoints) and b < len(keypoints):
                p1 = keypoints[a]
                p2 = keypoints[b]
                
                # Only draw if both points are confident
                if p1['confidence'] > 0.3 and p2['confidence'] > 0.3:
                    cv2.line(image, 
                            (p1['x'], p1['y']), 
                            (p2['x'], p2['y']),
                            (0, 255, 0), 2)
        
        # Draw keypoints
        for kp in keypoints:
            if kp['confidence'] > 0.3:
                cv2.circle(image, (kp['x'], kp['y']), 3, (255, 0, 0), -1)
    
    return image