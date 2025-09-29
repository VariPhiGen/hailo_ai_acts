import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path
import io
import av
import gc


class VideoClipRecorder:
    def __init__(self, maxlen=60, fps=20, prefix: str = "clips"):
        """
        Initialize video recorder for frame buffering and video generation.
        
        Args:
            maxlen: Maximum number of frames to buffer
            fps: Frames per second for video encoding
            prefix: Prefix for video filenames (used by external upload handlers)
        """
        self.prefix = prefix
        self.frame_buffer = deque(maxlen=maxlen)
        self.fps = fps
        
        print(f"Initiated VideoClipRecorder with buffer size {maxlen}")

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the RAM-backed buffer."""
        try:
            self.frame_buffer.append(frame.copy())
        except Exception as e:
            print(f"[ERROR] Failed to add frame to buffer: {e}")

    def generate_video_bytes(self, clear_after=True) -> bytes | None:
        """
        Memory-efficient in-memory MP4 (H.264 ultrafast) encoding.
        Takes a snapshot of the deque to avoid frame changes during encoding.
        """
        if not self.frame_buffer:
            return None

        try:
            # Snapshot the current frames to avoid mid-encode changes
            frame_buffer_copy = [frame.copy() for frame in list(self.frame_buffer)]
            h, w, _ = frame_buffer_copy[0].shape

            buf = io.BytesIO()
            container = av.open(buf, mode='w', format='mp4')

            stream = container.add_stream('libx264', rate=self.fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = 'yuv420p'

            codec_ctx = stream.codec_context
            codec_ctx.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile': 'baseline',
            }

            for frame_bgr in frame_buffer_copy:
                if frame_bgr.dtype != np.uint8:
                    frame_bgr = frame_bgr.astype(np.uint8)
                if not frame_bgr.flags['C_CONTIGUOUS']:
                    frame_bgr = np.ascontiguousarray(frame_bgr)

                #frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                video_frame = av.VideoFrame.from_ndarray(frame_bgr, format='rgb24')

                for packet in stream.encode(video_frame):
                    container.mux(packet)

                del video_frame, packet

            for packet in stream.encode():
                container.mux(packet)

            container.close()

            buf.seek(0)
            video_bytes = buf.read()
            buf.close()

            if clear_after:
                self.frame_buffer.clear()

            del container, stream, codec_ctx, frame_buffer_copy
            gc.collect()

            return video_bytes

        except Exception as e:
            print(f"[ERROR] Failed to generate video bytes: {e}")
            return None

    def save_images(self, org_img, save_dir: str, suffix: str):
        """Save a decoded OpenCV image (`numpy.ndarray`, dtype uint8, BGR)."""
        try:
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            org_path = os.path.join(save_dir, f"org_img_{timestamp}_{suffix}.jpg")

            if not isinstance(org_img, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray, got {type(org_img)}")

            if org_img.dtype != np.uint8:
                org_img = org_img.astype(np.uint8)

            if not org_img.flags['C_CONTIGUOUS']:
                org_img = np.ascontiguousarray(org_img)

            success = cv2.imwrite(org_path, org_img)
            print("Success")
            if not success:
                raise IOError(f"cv2.imwrite failed for image at {org_path}")

            return org_path

        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")
            return ""
