"""
anpr_handler.py  —  Hailo-side HTTP client for the ANPR inference server.
=========================================================================
  process_frame(image_crop, activity_name, extra_meta)
      → POST /detect-plate/    (multipart JPEG crop)
      → returns plate text from JSON: {"text": "MH12AB1234"}
      → persists result to  <sensor_id>_anpr.json

Set these in your .env file (same one used by detection.py):
    ANPR_SERVER_URL=https://192.168.1.50:8005
    # -- OR separately --
    ANPR_SERVER_IP=192.168.1.50
    ANPR_SERVER_PORT=8005
    ANPR_SERVER_SCHEME=https        # optional, defaults to https

load_dotenv() is called here so this module is fully self-contained
even if imported before detection.py has a chance to call it.
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
import urllib3

# Load .env file first so os.getenv() picks up the values.
# Safe to call multiple times — subsequent calls are no-ops.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv not installed — rely on os.environ being pre-populated

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

base_url = os.getenv("ANPR_SERVER_URL", "").strip().rstrip("/")


class ANPRHandler:
    """
    Thin HTTP wrapper around the ANPR Docker container.

    Always call process_frame() from a daemon thread — never from the
    GStreamer main callback.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        if not base_url:
            raise ValueError("ANPR_SERVER_URL not set")
        self._predict_url = f"https://{base_url}/detect-plate"

        # ── Circuit breaker ───────────────────────────────────────────────── #
        self.server_online  = True
        self.last_ping_time = 0.0
        self.ping_interval  = 10.0   # seconds between recovery attempts

        # ── Output JSON file ──────────────────────────────────────────────── #
        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "detections"
        )
        self._output_path = Path(f"{sensor_id}_anpr.json")

        # Print resolved URL at startup so misconfiguration is immediately obvious
        print(f"[ANPR] Handler ready.")
        print(f"[ANPR]   Endpoint : {self._predict_url}")
        print(f"[ANPR]   JSON log : {self._output_path}")

        if "localhost" in self._predict_url:
            print(
                "[ANPR] ⚠️  WARNING: endpoint resolved to localhost. "
                "If the ANPR server is remote, set ANPR_SERVER_URL (or "
                "ANPR_SERVER_IP + ANPR_SERVER_PORT) in your .env file."
            )

    # ════════════════════════════════════════════════════════════════════════ #
    #  PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════════ #

    def _encode_frame(self, image_input) -> bytes:
        """Converts a BGR numpy array (or bytes / path) to JPEG bytes."""
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
                raise ValueError("cv2.imencode failed")
            return buf.tobytes()

        elif isinstance(image_input, (bytes, bytearray)):
            arr   = np.frombuffer(bytes(image_input), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    return buf.tobytes()
            raise ValueError("Could not decode raw bytes as image")

        elif isinstance(image_input, (str, Path)):
            frame = cv2.imread(str(image_input))
            if frame is None:
                raise ValueError(f"Cannot read image: {image_input}")
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                return buf.tobytes()
            raise ValueError(f"cv2.imencode failed for: {image_input}")

        raise TypeError(f"Unsupported image type: {type(image_input)}")

    def _persist(self, record: dict):
        """Appends one detection record to <sensor_id>_anpr.json."""
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
        except IOError as exc:
            logging.error(f"[ANPR] Could not write {self._output_path}: {exc}")

    # ── Circuit breaker ───────────────────────────────────────────────────── #

    def _circuit_open(self) -> bool:
        if self.server_online:
            return False

        now = time.time()
        if (now - self.last_ping_time) < self.ping_interval:
            return True

        # Blackout window expired — optimistically allow one real attempt through.
        # If it fails again, _trip_breaker() will be called and reset the timer.
        logging.info("[ANPR] Blackout window expired — allowing retry.")
        self.server_online = True
        return False

    def _trip_breaker(self, reason: str = ""):
        logging.warning(
            f"[ANPR] Server OFFLINE — {reason}. "
            f"Next retry in {self.ping_interval}s."
        )
        self.server_online  = False
        self.last_ping_time = time.time()

    # ════════════════════════════════════════════════════════════════════════ #
    #  PUBLIC INFERENCE METHOD
    # ════════════════════════════════════════════════════════════════════════ #

    def process_frame(
        self,
        image_crop,
        activity_name: str = "unknown_activity",
        extra_meta: dict | None = None,
        timeout: float = 10.0,
    ) -> str | None:
        """
        POSTs a JPEG crop to /detect-plate/ and returns the plate string or None.
        Expected server response: {"text": "MH12AB1234"}

        Always call from a daemon thread — never from the GStreamer callback.
        """
        if self._circuit_open():
            return None

        try:
            jpeg_bytes = self._encode_frame(image_crop)
        except (ValueError, TypeError) as exc:
            logging.error(f"[ANPR] Frame encode failed: {exc}")
            return None

        try:
            resp = requests.post(
                self._predict_url,
                files={"file": ("crop.jpg", jpeg_bytes, "image/jpeg")},
                timeout=timeout,
                verify=False,
            )
            resp.raise_for_status()

            try:
                plate_text = resp.json().get("text", "").strip()
            except (ValueError, KeyError):
                logging.error(f"[ANPR] Bad JSON: {resp.text[:300]}")
                return None

            if not plate_text:
                logging.info("[ANPR] Empty plate in response.")
                return None

        except requests.exceptions.ConnectionError as exc:
            self._trip_breaker(reason=str(exc))
            return None
        except requests.exceptions.Timeout:
            logging.error(f"[ANPR] Timed out after {timeout}s.")
            return None
        except requests.exceptions.HTTPError as exc:
            logging.error(
                f"[ANPR] HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            )
            return None
        except Exception as exc:
            logging.error(f"[ANPR] Unexpected error: {exc}")
            return None

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "activity":  activity_name,
            "plate":     plate_text,
            **(extra_meta or {}),
        }
        self._persist(record)
        print(f"[ANPR] ✅ Plate: '{plate_text}' | {extra_meta}")
        return plate_text
