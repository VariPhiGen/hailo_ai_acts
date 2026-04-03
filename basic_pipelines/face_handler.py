"""
face_recognition_handler.py  —  Hailo-side HTTP client for the Face Recognition server.
=========================================================================================
  match_faces(face_crops)
      → POST /image_match/   (multipart — multiple "images" fields in ONE request)
      → returns list of per-crop result dicts:
            {"tracker_id": int, "status": "match_found"|"no_match",
             "person_name": str|None, "confidence": float|None}
      → persists results to <sensor_id>_facerec.json

  match_face(crop, tracker_id, extra_meta)          ← single-image convenience wrapper

Key differences from ANPRHandler
─────────────────────────────────
• Batch API: all face crops from a frame are sent in ONE POST for efficiency.
• Response is a JSON array, indexed by image_name we embed in the filename.
• verify=False kept because internal servers commonly use self-signed certs.

Set in .env (same file detection.py loads):
    FACEREC_SERVER_URL=http://192.168.1.50:8006/image_match/
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


base_url = os.getenv("FACEREC_SERVER_URL", "").strip().rstrip("/")


class FaceRecognitionHandler:
    """
    HTTP wrapper around the Face Recognition Docker container.

    Always call match_faces() / match_face() from a daemon thread —
    never directly from the GStreamer main callback.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        
        if not base_url:
            raise ValueError("FACEREC_SERVER_URL not set")

        self._predict_url = f"{base_url}/image_match"

        # ── Circuit breaker ───────────────────────────────────────────────── #
        self.server_online  = True
        self.last_ping_time = 0.0
        self.ping_interval  = 10.0

        # ── Output file ───────────────────────────────────────────────────── #
        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "detections"
        )
        self._output_path = Path(f"{sensor_id}_facerec.json")

        print(f"[FaceRec] Handler ready.")
        print(f"[FaceRec]   Endpoint : {self._predict_url}")
        print(f"[FaceRec]   JSON log : {self._output_path}")

        if "localhost" in self._predict_url:
            print(
                "[FaceRec] ⚠️  Endpoint resolved to localhost. "
                "Set FACEREC_SERVER_URL in your .env if the server is remote."
            )

    # ════════════════════════════════════════════════════════════════════════ #
    #  PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════════ #

    def _encode_frame(self, image: np.ndarray) -> bytes:
        """Converts a BGR numpy array to JPEG bytes."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return buf.tobytes()

    def _persist(self, records: list):
        """Appends a list of recognition results to <sensor_id>_facerec.json."""
        existing: list = []
        if self._output_path.exists():
            try:
                with open(self._output_path, "r") as fh:
                    data = json.load(fh)
                    existing = data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, IOError):
                existing = []
        existing.extend(records)
        try:
            with open(self._output_path, "w") as fh:
                json.dump(existing, fh, indent=2)
        except IOError as exc:
            logging.error(f"[FaceRec] Could not write {self._output_path}: {exc}")

    # ── Circuit breaker ───────────────────────────────────────────────────── #

    def _circuit_open(self) -> bool:
        if self.server_online:
            return False

        now = time.time()
        if (now - self.last_ping_time) < self.ping_interval:
            return True

        # Blackout window expired — allow one real attempt through.
        # If it fails again _trip_breaker resets the timer.
        logging.info("[FaceRec] Blackout expired — allowing retry.")
        self.server_online = True
        return False

    def _trip_breaker(self, reason: str = ""):
        logging.warning(
            f"[FaceRec] Server OFFLINE — {reason}. "
            f"Next retry in {self.ping_interval}s."
        )
        self.server_online  = False
        self.last_ping_time = time.time()

    # ════════════════════════════════════════════════════════════════════════ #
    #  PUBLIC INFERENCE METHODS
    # ════════════════════════════════════════════════════════════════════════ #

    def match_faces(
        self,
        face_crops: list[dict],
        activity_name: str = "FaceRecognition",
        timeout: float = 15.0,
    ) -> list[dict]:
        """
        Sends multiple face crops to the API in ONE batch POST.

        Parameters
        ----------
        face_crops : list of dicts, each with:
                       "tracker_id" : int   — Hailo tracker ID
                       "crop"       : ndarray — BGR face image
                       "extra_meta" : dict  — optional extra fields to log
        activity_name : for the JSON log record
        timeout       : HTTP timeout in seconds

        Returns
        -------
        List of result dicts:
            {
              "tracker_id"  : int,
              "status"      : "match_found" | "no_match" | "error",
              "person_name" : str | None,
              "confidence"  : float | None,
            }
        On total failure returns an empty list.
        """
        if not face_crops:
            return []

        if self._circuit_open():
            return []

        # ── Encode all crops and build multipart fields ───────────────────── #
        # We embed the tracker_id in the filename so we can map results back.
        # Filename format:  tid_{tracker_id}.jpg
        files  = []
        id_map = {}   # image_name → tracker_id for reverse lookup

        for item in face_crops:
            tid  = item["tracker_id"]
            crop = item.get("crop")
            if crop is None or crop.size == 0:
                continue
            try:
                jpeg = self._encode_frame(crop)
            except Exception as exc:
                logging.error(f"[FaceRec] Encode failed tid={tid}: {exc}")
                continue

            img_name = f"tid_{tid}.jpg"
            id_map[img_name] = tid
            # requests multipart: list of ("images", (filename, data, mime))
            files.append(("images", (img_name, jpeg, "image/jpeg")))

        if not files:
            return []

        # ── POST ──────────────────────────────────────────────────────────── #
        try:
            resp = requests.post(
                self._predict_url,
                files=files,
                timeout=timeout,
                verify=False,
            )
            resp.raise_for_status()
            api_data = resp.json()

        except requests.exceptions.ConnectionError as exc:
            self._trip_breaker(reason=str(exc))
            return []
        except requests.exceptions.Timeout:
            logging.error(f"[FaceRec] Timed out after {timeout}s.")
            return []
        except requests.exceptions.HTTPError as exc:
            logging.error(
                f"[FaceRec] HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            )
            return []
        except Exception as exc:
            logging.error(f"[FaceRec] Unexpected error: {exc}")
            return []

        # ── Parse response ────────────────────────────────────────────────── #
        # API response shape:
        # {"total_images": N, "results": [
        #     {"image_name": "tid_42.jpg", "status": "match_found",
        #      "person_name": "Alice", "confidence": 97.8},
        #     {"image_name": "tid_7.jpg",  "status": "no_match"},
        #     ...
        # ]}
        output   = []
        now_iso  = datetime.utcnow().isoformat() + "Z"
        to_persist = []

        raw_results = api_data.get("results", [])

        # Build a lookup by image_name for O(1) access
        api_lookup = {r.get("image_name", ""): r for r in raw_results}

        for img_name, tid in id_map.items():
            raw = api_lookup.get(img_name, {})
            status      = raw.get("status", "no_match")
            person_name = raw.get("person_name") if status == "match_found" else None
            confidence  = raw.get("confidence")  if status == "match_found" else None

            result = {
                "tracker_id":  tid,
                "status":      status,
                "person_name": person_name,
                "confidence":  confidence,
            }
            output.append(result)

            # Find matching item for extra_meta
            extra_meta = next(
                (item.get("extra_meta", {}) for item in face_crops
                 if item["tracker_id"] == tid),
                {}
            )

            record = {
                "timestamp":   now_iso,
                "activity":    activity_name,
                "tracker_id":  tid,
                "status":      status,
                "person_name": person_name,
                "confidence":  confidence,
                **(extra_meta or {}),
            }
            to_persist.append(record)

            if status == "match_found":
                print(
                    f"[FaceRec] ✅ Match — tid={tid} → '{person_name}' "
                    f"({confidence}%)"
                )
            else:
                logging.info(f"[FaceRec] No match — tid={tid}")

        if to_persist:
            self._persist(to_persist)

        return output

    def match_face(
        self,
        crop: np.ndarray,
        tracker_id: int,
        activity_name: str = "FaceRecognition",
        extra_meta: dict | None = None,
        timeout: float = 10.0,
    ) -> dict | None:
        """
        Single-image convenience wrapper around match_faces().
        Returns one result dict or None on failure.
        """
        results = self.match_faces(
            [{"tracker_id": tracker_id, "crop": crop, "extra_meta": extra_meta}],
            activity_name=activity_name,
            timeout=timeout,
        )
        return results[0] if results else None
