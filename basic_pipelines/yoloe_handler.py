import cv2
import numpy as np
import requests
import json
import logging
from pathlib import Path
import os
import time  # <-- Added for the circuit breaker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SERVER_IP = os.getenv("YOLOE_SERVER_IP", "localhost")
SERVER_PORT = os.getenv("YOLOE_SERVER_PORT", "5000")
YOLOE_SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}/predict_prompt"


class YOLOEHandler:
    def __init__(self, config=None):
        self.config = config or {}
        self.results_queue = None

        # --- Circuit Breaker State ---
        self.server_online = True
        self.last_ping_time = 0
        self.ping_interval = 5.0  # Wait 5 seconds before checking if server is back
        # -----------------------------

        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "detections"
        )
        self._output_path = Path(f"{sensor_id}.json")
        logging.info(f"YOLOEHandler ready — detections will be saved to '{self._output_path}'")

    def set_results_queue(self, queue):
        self.results_queue = queue

    def _encode_frame(self, image_input) -> bytes:
        # [KEEP YOUR EXISTING _encode_frame CODE HERE EXACTLY AS IT WAS]
        if isinstance(image_input, np.ndarray):
            frame = image_input
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)

            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError("Failed to encode numpy array to JPEG.")
            return buf.tobytes()

        elif isinstance(image_input, (bytes, bytearray)):
            raw = bytes(image_input)
            np_arr = np.frombuffer(raw, dtype=np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                success, buf = cv2.imencode(".jpg", frame)
                if success:
                    return buf.tobytes()
            raise ValueError("Could not decode bytes as any known image format.")

        elif isinstance(image_input, (str, Path)):
            frame = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                return buf.tobytes()
            raise ValueError(f"Failed to encode '{image_input}' to JPEG.")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    def _append_to_json(self, record: dict):
        # [KEEP YOUR EXISTING _append_to_json CODE HERE]
        records = []
        if self._output_path.exists():
            try:
                with open(self._output_path, "r") as f:
                    records = json.load(f)
                if not isinstance(records, list):
                    records = [records]
            except (json.JSONDecodeError, IOError):
                records = []

        records.append(record)
        with open(self._output_path, "w") as f:
            json.dump(records, f, indent=2)

    def text_prompt(self, image_input, prompts, conf=0.05, activity_uses_yoloe=True):
        """
        Main inference call. Added `activity_uses_yoloe` to easily bypass.
        """
        # 1. SCENARIO A: Bypass instantly if the activity doesn't use YOLOE
        if not activity_uses_yoloe or not prompts:
            return None

        # 2. SCENARIO B: Circuit Breaker for offline server
        current_time = time.time()
        if not self.server_online:
            # If 5 seconds haven't passed yet, bypass instantly without blocking Hailo
            if (current_time - self.last_ping_time) < self.ping_interval:
                return None 
            
            # 5 seconds passed. Do a lightning-fast (1 second) ping to see if it's awake
            self.last_ping_time = current_time
            try:
                ping_url = YOLOE_SERVER_URL.replace("/predict_prompt", "/docs")
                response = requests.get(ping_url, timeout=1.0)
                if response.status_code == 200:
                    logging.info("YOLOE server is back ONLINE! Resuming inferences.")
                    self.server_online = True
            except requests.exceptions.RequestException:
                # Still offline. Bypass again.
                return None

        # 3. Server is online. Proceed with normal inference.
        try:
            encoded_bytes = self._encode_frame(image_input)
        except (ValueError, TypeError) as e:
            logging.error(f"Frame encoding failed: {e}")
            return None

        prompts_str = ",".join(str(p).strip() for p in prompts) if isinstance(prompts, list) else str(prompts)
        files = {"file": ("frame.jpg", encoded_bytes, "image/jpeg")}
        data  = {"prompts": prompts_str, "conf": float(conf)}

        try:
            response = requests.post(YOLOE_SERVER_URL, files=files, data=data, timeout=120.0)
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            # Server just died! Trip the circuit breaker and bypass.
            logging.warning("YOLOE Server went OFFLINE. Bypassing YOLOE to maintain Hailo speed.")
            self.server_online = False
            self.last_ping_time = time.time()
            return None
            
        except requests.exceptions.Timeout:
            logging.error("YOLOE Server timed out — model may still be loading.")
            return None
        except requests.exceptions.HTTPError:
            logging.error(f"Server HTTP error {response.status_code}: {response.text[:200]}")
            return None
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return None

        # Persist to disk and queue
        self._append_to_json(result)
        if self.results_queue is not None:
            try:
                self.results_queue.put_nowait(result)
            except Exception:
                pass

        return result
