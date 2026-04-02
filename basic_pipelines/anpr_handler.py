"""
anpr_handler.py  —  Hailo-side HTTP client for the ANPR inference server.
=========================================================================
Mirrors the YOLOEHandler architectural pattern exactly:

  process_frame(image_crop, activity_name, extra_meta)
      → POST /process_frame/   (multipart JPEG crop)
      → returns plain-text licence plate string, e.g. "MH12AB1234"
      → persists result to  <sensor_id>_anpr.json

Circuit breaker: if the server goes offline the call is skipped
instantly — the Hailo GStreamer pipeline never blocks waiting for ANPR.
"""

import cv2
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Server address (override via env vars in production) ─────────────────── #
ANPR_SERVER_IP   = os.getenv("ANPR_SERVER_IP",   "localhost")
ANPR_SERVER_PORT = os.getenv("ANPR_SERVER_PORT",  "3001")
_BASE            = f"https://{ANPR_SERVER_IP}:{ANPR_SERVER_PORT}"

ANPR_PREDICT_URL = f"{_BASE}/process_frame/"
ANPR_HEALTH_URL  = f"{_BASE}/health"          # used for recovery pings


class ANPRHandler:
    """
    Thin HTTP wrapper around the ANPR Docker container.

    Usage (event-driven, called from a one-off daemon thread):
        handler = ANPRHandler(config)
        plate = handler.process_frame(crop_bgr, "AnprTest", {"tracker_id": 42})
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # ── Circuit breaker state ─────────────────────────────────────────── #
        self.server_online  = True    # optimistic: assume server is up at start
        self.last_ping_time = 0.0
        self.ping_interval  = 5.0    # seconds between recovery pings

        # ── Output file: <sensor_id>_anpr.json ───────────────────────────── #
        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "detections"
        )
        self._output_path = Path(f"{sensor_id}_anpr.json")
        logging.info(
            f"ANPRHandler ready — results will be saved to '{self._output_path}'"
        )

    # ════════════════════════════════════════════════════════════════════════ #
    #  PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════════ #

    def _encode_frame(self, image_input) -> bytes:
        """
        Safely converts a raw BGR numpy array (or bytes/path) into JPEG bytes.
        Mirrors YOLOEHandler._encode_frame exactly for consistency.
        """
        if isinstance(image_input, np.ndarray):
            frame = image_input
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Normalise channel count → BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("cv2.imencode failed — could not convert frame to JPEG")
            return buf.tobytes()

        elif isinstance(image_input, (bytes, bytearray)):
            arr   = np.frombuffer(bytes(image_input), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    return buf.tobytes()
            raise ValueError("Could not decode raw bytes as an image")

        elif isinstance(image_input, (str, Path)):
            frame = cv2.imread(str(image_input))
            if frame is None:
                raise ValueError(f"Cannot read image from path: {image_input}")
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                return buf.tobytes()
            raise ValueError(f"cv2.imencode failed for path: {image_input}")

        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # ── JSON persistence ──────────────────────────────────────────────────── #

    def _persist(self, record: dict):
        """Appends a detection record to <sensor_id>_anpr.json."""
        records: list = []
        if self._output_path.exists():
            try:
                with open(self._output_path, "r") as fh:
                    data = json.load(fh)
                    records = data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, IOError):
                records = []

        records.append(record)

        try:
            with open(self._output_path, "w") as fh:
                json.dump(records, fh, indent=2)
        except IOError as e:
            logging.error(f"[ANPR] Could not write to {self._output_path}: {e}")

    # ── Circuit breaker ───────────────────────────────────────────────────── #

    def _circuit_open(self) -> bool:
        """
        Returns True  → skip the HTTP call (server offline, still in blackout).
        Returns False → allow the call (server online or recovery ping succeeded).
        """
        if self.server_online:
            return False   # fast path: server is healthy

        now = time.time()
        if (now - self.last_ping_time) < self.ping_interval:
            return True    # still inside the blackout window

        # ── Recovery ping ────────────────────────────────────────────────── #
        self.last_ping_time = now
        try:
            r = requests.get(ANPR_HEALTH_URL, timeout=1.0, verify=False)
            if r.status_code == 200:
                logging.info("[ANPR] Server back ONLINE — resuming inference.")
                self.server_online = True
                return False
        except requests.exceptions.RequestException:
            pass

        return True   # still offline

    def _trip_breaker(self):
        """Called when a connection error is detected; marks server offline."""
        logging.warning(
            "[ANPR] Server OFFLINE — bypassing calls to protect the Hailo pipeline."
        )
        self.server_online  = False
        self.last_ping_time = time.time()

    # ════════════════════════════════════════════════════════════════════════ #
    #  PUBLIC INFERENCE METHOD
    # ════════════════════════════════════════════════════════════════════════ #

    def process_frame(
        self,
        image_crop,
        activity_name: str = "unkown_activity",
        extra_meta: dict | None = None,
        timeout: float = 10.0,
    ) -> str | None:
        """
        POSTs a JPEG crop to POST /process_frame/ and returns the licence-plate
        string (e.g. "MH12AB1234") or None on any failure.

        Parameters
        ----------
        image_crop    : BGR numpy array (or bytes / path) — the vehicle crop.
        activity_name : Name of the calling activity, used in the log record.
        extra_meta    : Any extra key/values to embed in the persisted record
                        (e.g. tracker_id, zone_name, dwell_seconds).
        timeout       : HTTP request timeout in seconds.

        Thread safety
        -------------
        Designed to be called from a one-off daemon thread so the GStreamer
        main thread is never blocked.
        """
        # ── Guard: circuit open → skip immediately ────────────────────────── #
        if self._circuit_open():
            return None

        # ── Encode frame → JPEG bytes ─────────────────────────────────────── #
        try:
            jpeg_bytes = self._encode_frame(image_crop)
        except (ValueError, TypeError) as exc:
            logging.error(f"[ANPR] Frame encode failed: {exc}")
            return None

        # ── POST to ANPR server ───────────────────────────────────────────── #
        try:
            resp = requests.post(
                ANPR_PREDICT_URL,
                files={"file": ("crop.jpg", jpeg_bytes, "image/jpeg")},
                timeout=timeout,
                verify=False,   # self-signed cert inside Docker is common
            )
            resp.raise_for_status()

            # Response is plain text: "MH12AB1234"
            plate_text = resp.text.strip()

        except requests.exceptions.ConnectionError:
            self._trip_breaker()
            return None
        except requests.exceptions.Timeout:
            logging.error(f"[ANPR] POST {ANPR_PREDICT_URL} timed out.")
            return None
        except requests.exceptions.HTTPError as exc:
            logging.error(
                f"[ANPR] HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            )
            return None
        except Exception as exc:
            logging.error(f"[ANPR] Unexpected error: {exc}")
            return None

        # ── Persist result ────────────────────────────────────────────────── #
        record = {
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "activity":      activity_name,
            "plate":         plate_text,
            **(extra_meta or {}),
        }
        self._persist(record)

        logging.info(f"[ANPR] Detected plate: '{plate_text}' — meta={extra_meta}")
        return plate_text