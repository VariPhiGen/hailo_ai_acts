"""
anpr_handler.py  —  ANPR Inference Handler  (Hailo-side)
=========================================================
Mirrors yoloe_handler.py exactly:
  - Circuit breaker so a dead ANPR server never blocks the Hailo pipeline
  - Encodes any image type (numpy / bytes / path) to JPEG
  - POSTs a CROPPED vehicle image to the ANPR endpoint
  - Returns the plate string (or None if server is offline / no plate found)
  - Appends every result to a JSON log on disk

ANPR server endpoint:
  POST https://localhost/process_frame/
  -F "file=@frame.jpg"
  Response: plain text  e.g.  "MH12AB1234"

Environment variables (optional):
  ANPR_SERVER_IP    default: localhost
  ANPR_SERVER_PORT  default: 443
  ANPR_USE_HTTPS    default: 1   (set 0 for plain http)
"""

import cv2
import numpy as np
import requests
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ANPR_SERVER_IP   = os.getenv("ANPR_SERVER_IP",   "localhost")
ANPR_SERVER_PORT = os.getenv("ANPR_SERVER_PORT",  "443")
_use_https       = os.getenv("ANPR_USE_HTTPS", "1") != "0"
_scheme          = "https" if _use_https else "http"
ANPR_SERVER_URL  = f"{_scheme}://{ANPR_SERVER_IP}:{ANPR_SERVER_PORT}/process_frame/"


class ANPRHandler:
    def __init__(self, config=None):
        self.config = config or {}

        # ── Circuit Breaker State (mirrors yoloe_handler) ─────────────────
        self.server_online  = True
        self.last_ping_time = 0.0
        self.ping_interval  = 5.0   # seconds between retry pings when offline
        # ──────────────────────────────────────────────────────────────────

        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "anpr"
        )
        self._output_path = Path(f"{sensor_id}_anpr_log.json")
        logging.info(f"ANPRHandler ready — log: '{self._output_path}'")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _encode_frame(self, image_input) -> bytes:
        """Encode any supported image input to JPEG bytes."""
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
            np_arr = np.frombuffer(bytes(image_input), dtype=np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                success, buf = cv2.imencode(".jpg", frame)
                if success:
                    return buf.tobytes()
            raise ValueError("Could not decode bytes as a known image format.")

        elif isinstance(image_input, (str, Path)):
            frame = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read image file: '{image_input}'")
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                return buf.tobytes()
            raise ValueError(f"Failed to encode '{image_input}' to JPEG.")

        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    def _append_to_json(self, record: dict):
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

    # ── Public API ────────────────────────────────────────────────────────

    def get_plate(
        self,
        image_input,
        activity_name: str = "unknown",
        extra_meta: dict   = None,
    ) -> str | None:
        """
        Send a cropped vehicle image to the ANPR server.
        Returns plate text (str) or None if server offline / no plate found.
        Safe to call from a background thread.
        """
        # ── Circuit Breaker ──────────────────────────────────────────────
        current_time = time.time()
        if not self.server_online:
            if (current_time - self.last_ping_time) < self.ping_interval:
                return None
            self.last_ping_time = current_time
            try:
                ping_url = f"{_scheme}://{ANPR_SERVER_IP}:{ANPR_SERVER_PORT}/"
                resp = requests.get(ping_url, timeout=1.0, verify=False)
                if resp.status_code < 500:
                    logging.info("ANPR server is back ONLINE!")
                    self.server_online = True
                else:
                    return None
            except requests.exceptions.RequestException:
                return None

        # ── Encode ───────────────────────────────────────────────────────
        try:
            jpeg_bytes = self._encode_frame(image_input)
        except (ValueError, TypeError) as e:
            logging.error(f"[ANPR] Frame encoding failed: {e}")
            return None

        # ── POST ─────────────────────────────────────────────────────────
        files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
        try:
            response = requests.post(
                ANPR_SERVER_URL, files=files, timeout=30.0, verify=False
            )
            response.raise_for_status()

        except requests.exceptions.ConnectionError:
            logging.warning(
                "[ANPR] Server went OFFLINE. Bypassing ANPR to maintain pipeline speed."
            )
            self.server_online = False
            self.last_ping_time = time.time()
            return None

        except requests.exceptions.Timeout:
            logging.error("[ANPR] Server timed out.")
            return None

        except requests.exceptions.HTTPError:
            logging.error(
                f"[ANPR] HTTP error {response.status_code}: {response.text[:200]}"
            )
            return None

        except Exception as e:
            logging.error(f"[ANPR] Request failed: {e}")
            return None

        # ── Parse plain-text response ─────────────────────────────────────
        plate_text = response.text.strip()
        if not plate_text:
            return None

        logging.info(f"[ANPR] Plate: '{plate_text}'  activity='{activity_name}'")

        # ── Persist to disk ───────────────────────────────────────────────
        record = {
            "plate":     plate_text,
            "activity":  activity_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if extra_meta:
            record.update(extra_meta)
        self._append_to_json(record)
        return plate_text
