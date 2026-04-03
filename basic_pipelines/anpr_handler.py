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

# ── Minimal 1×1 white JPEG for recovery pings ────────────────────────────── #
_PING_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\x1f&+2++\x1f\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
    b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
    b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n"
    b"\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
    b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
    b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
    b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
    b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
    b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00"
    b"\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"
)


def _resolve_predict_url() -> str:
    """
    Builds the full predict URL from env vars.

    Priority:
      1. ANPR_SERVER_URL  (complete URL, e.g. https://1.2.3.4:8005)
      2. ANPR_SERVER_SCHEME + ANPR_SERVER_IP + ANPR_SERVER_PORT
      3. Hardcoded localhost:8005 fallback (always wrong in production —
         if you see this URL printed at startup, check your .env file)
    """
    full_url = os.getenv("ANPR_SERVER_URL", "").strip().rstrip("/")
    if full_url:
        return f"{full_url}/detect-plate"
        
    ip     = os.getenv("ANPR_SERVER_IP",     "localhost").strip()
    return f"https://{ip}/detect-plate"


class ANPRHandler:
    """
    Thin HTTP wrapper around the ANPR Docker container.

    Always call process_frame() from a daemon thread — never from the
    GStreamer main callback.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # URL is resolved here (after load_dotenv above) so env vars are live.
        self._predict_url = _resolve_predict_url()

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

        # Recovery: POST a 1×1 JPEG — any HTTP response means TCP is alive
        self.last_ping_time = now
        logging.info(f"[ANPR] Recovery ping → {self._predict_url}")
        try:
            r = requests.post(
                self._predict_url,
                files={"file": ("ping.jpg", _PING_JPEG, "image/jpeg")},
                timeout=3.0,
                verify=False,
            )
            logging.info(f"[ANPR] Server ONLINE again (HTTP {r.status_code}).")
            self.server_online = True
            return False
        except requests.exceptions.ConnectionError as exc:
            logging.warning(f"[ANPR] Recovery ping failed: {exc}")
        except requests.exceptions.Timeout:
            logging.warning("[ANPR] Recovery ping timed out.")
        except Exception as exc:
            logging.warning(f"[ANPR] Recovery ping error: {exc}")
        return True

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
