"""
yoloe_handler.py  —  Hailo-side HTTP client for the YOLOE inference server.
============================================================================
Three inference methods, all protected by the same circuit breaker:
  text_prompt()    — POST /predict_prompt   (frame + text labels)
  visual_prompt()  — POST /predict_visual   (frame + activity_id)
  both_prompt()    — POST /predict_both     (frame + labels + activity_id)

The visual prompt reference image and annotations live entirely on the
server side (in visual_prompts.json managed by app.py).
The Hailo side never needs to send or know about reference images.
"""

import cv2
import numpy as np
import requests
import json
import logging
from pathlib import Path
import os
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SERVER_IP   = os.getenv("YOLOE_SERVER_IP",   "localhost")
SERVER_PORT = os.getenv("YOLOE_SERVER_PORT",  "5000")
_BASE       = f"http://{SERVER_IP}:{SERVER_PORT}"

PREDICT_TEXT_URL   = f"{_BASE}/predict_prompt"
PREDICT_VISUAL_URL = f"{_BASE}/predict_visual"
PREDICT_BOTH_URL   = f"{_BASE}/predict_both"
HEALTH_URL         = f"{_BASE}/health"


class YOLOEHandler:

    def __init__(self, config=None):
        self.config       = config or {}
        self.results_queue = None

        # ── Circuit breaker ───────────────────────────────────────────────── #
        self.server_online  = True
        self.last_ping_time = 0.0
        self.ping_interval  = 5.0   # seconds between recovery pings

        # ── Output file ───────────────────────────────────────────────────── #
        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "detections"
        )
        self._output_path = Path(f"{sensor_id}_yoloe.json")
        logging.info(f"YOLOEHandler ready — saving detections to '{self._output_path}'")

    # ── Queue ─────────────────────────────────────────────────────────────── #

    def set_results_queue(self, queue):
        self.results_queue = queue

    # ── Frame encoding ────────────────────────────────────────────────────── #

    def _encode_frame(self, image_input) -> bytes:
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
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Failed to encode frame to JPEG")
            return buf.tobytes()

        elif isinstance(image_input, (bytes, bytearray)):
            arr   = np.frombuffer(bytes(image_input), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    return buf.tobytes()
            raise ValueError("Could not decode bytes as image")

        elif isinstance(image_input, (str, Path)):
            frame = cv2.imread(str(image_input))
            if frame is None:
                raise ValueError(f"Cannot read: {image_input}")
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                return buf.tobytes()
            raise ValueError(f"Failed to encode '{image_input}' to JPEG")

        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # ── JSON persistence + queue ──────────────────────────────────────────── #

    def _persist_and_queue(self, result: dict):
        # Append to sensor_id.json
        records = []
        if self._output_path.exists():
            try:
                with open(self._output_path, "r") as f:
                    records = json.load(f)
                if not isinstance(records, list):
                    records = [records]
            except (json.JSONDecodeError, IOError):
                records = []
        records.append(result)
        with open(self._output_path, "w") as f:
            json.dump(records, f, indent=2)

        # Push to Kafka queue if connected
        if self.results_queue is not None:
            try:
                self.results_queue.put_nowait(result)
            except Exception:
                pass

    # ── Circuit breaker ───────────────────────────────────────────────────── #

    def _circuit_open(self) -> bool:
        """
        Returns True  → block the call (server known-offline, still in blackout)
        Returns False → allow the call (server online or recovery ping succeeded)
        """
        if self.server_online:
            return False

        now = time.time()
        if (now - self.last_ping_time) < self.ping_interval:
            return True   # still in blackout window

        # Recovery ping
        self.last_ping_time = now
        try:
            r = requests.get(HEALTH_URL, timeout=1.0)
            if r.status_code == 200:
                logging.info("YOLOE server back ONLINE — resuming.")
                self.server_online = True
                return False
        except requests.exceptions.RequestException:
            pass
        return True       # still offline

    def _trip_breaker(self):
        logging.warning(
            "YOLOE server OFFLINE — bypassing to protect Hailo pipeline speed."
        )
        self.server_online  = False
        self.last_ping_time = time.time()

    # ── Shared POST ───────────────────────────────────────────────────────── #

    def _post(self, url: str, files: dict, data: dict,
              timeout: float = 120.0) -> dict | None:
        try:
            resp = requests.post(url, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            self._persist_and_queue(result)
            return result

        except requests.exceptions.ConnectionError:
            self._trip_breaker()
            return None
        except requests.exceptions.Timeout:
            logging.error(f"POST {url} timed out — model may still be loading.")
            return None
        except requests.exceptions.HTTPError as e:
            logging.error(
                f"POST {url}  HTTP {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
            return None
        except Exception as e:
            logging.error(f"POST {url} failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════ #
    #  PUBLIC INFERENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════ #

    def text_prompt(
        self,
        image_input,
        prompts,
        conf: float = 0.05,
        activity_uses_yoloe: bool = True,
    ) -> dict | None:
        """
        Text-prompt inference  →  POST /predict_prompt
        prompts: list of str  or  comma-separated str
        """
        if not activity_uses_yoloe or not prompts:
            return None
        if self._circuit_open():
            return None

        try:
            encoded = self._encode_frame(image_input)
        except (ValueError, TypeError) as e:
            logging.error(f"Frame encode failed: {e}")
            return None

        prompts_str = (
            ",".join(str(p).strip() for p in prompts)
            if isinstance(prompts, list) else str(prompts)
        )
        return self._post(
            PREDICT_TEXT_URL,
            files={"file": ("frame.jpg", encoded, "image/jpeg")},
            data ={"prompts": prompts_str, "conf": float(conf)},
        )

    def visual_prompt(
        self,
        image_input,
        activity_id: str,
        conf: float = 0.05,
    ) -> dict | None:
        """
        Visual-prompt inference  →  POST /predict_visual
        The server reads reference image + annotations from visual_prompts.json.
        The Hailo side only needs to pass the activity_id.
        Returns None if the server returns 404 (activity not annotated yet).
        """
        if not activity_id:
            return None
        if self._circuit_open():
            return None

        try:
            encoded = self._encode_frame(image_input)
        except (ValueError, TypeError) as e:
            logging.error(f"Frame encode failed: {e}")
            return None

        return self._post(
            PREDICT_VISUAL_URL,
            files={"file": ("frame.jpg", encoded, "image/jpeg")},
            data ={"activity_id": activity_id, "conf": float(conf)},
        )

    def both_prompt(
        self,
        image_input,
        prompts,
        activity_id: str,
        conf: float = 0.05,
    ) -> dict | None:
        """
        Combined text + visual inference  →  POST /predict_both
        prompts     : text labels for the text pass
        activity_id : activity name with visual prompts saved on the server
        """
        if not prompts or not activity_id:
            return None
        if self._circuit_open():
            return None

        try:
            encoded = self._encode_frame(image_input)
        except (ValueError, TypeError) as e:
            logging.error(f"Frame encode failed: {e}")
            return None

        prompts_str = (
            ",".join(str(p).strip() for p in prompts)
            if isinstance(prompts, list) else str(prompts)
        )
        return self._post(
            PREDICT_BOTH_URL,
            files={"file": ("frame.jpg", encoded, "image/jpeg")},
            data ={"prompts": prompts_str,
                   "activity_id": activity_id,
                   "conf": float(conf)},
        )
