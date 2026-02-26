import cv2
import numpy as np
import requests
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

YOLOE_SERVER_URL = "http://localhost:5000/predict_prompt"


class YOLOEHandler:
    def __init__(self, config=None):
        """
        config: Full configuration.json dictionary loaded by detection.py.

        The output JSON file is named after sensor_id or camera_id from config,
        e.g. "cam_01.json". Falls back to "detections.json" if neither is set.
        """
        self.config = config or {}
        self.results_queue = None

        sensor_id = (
            self.config.get("sensor_id")
            or self.config.get("camera_details", {}).get("camera_id")
            or self.config.get("camera_id")
            or "detections"
        )
        self._output_path = Path(f"{sensor_id}.json")
        logging.info(f"YOLOEHandler ready — detections will be saved to '{self._output_path}'")

    def set_results_queue(self, queue):
        """Pass results into a queue for downstream consumers (e.g. Kafka)."""
        self.results_queue = queue

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _encode_frame(self, image_input) -> bytes:
        """
        Convert any image input into JPEG bytes for the POST request.

        Accepted types (mirrors what the hailo pipeline can produce):
          - numpy.ndarray  : any dtype, grayscale / BGR / BGRA
          - bytes          : already-encoded (JPEG, PNG, BMP, TIFF, WEBP, ...)
                             OR raw numpy .tobytes() dump
          - bytearray      : same as bytes
          - str / Path     : file path — read and encode to JPEG
        """
        # ---- numpy array -----------------------------------------------
        if isinstance(image_input, np.ndarray):
            frame = image_input

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            if frame.ndim == 2:                              # grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:   # BGRA / RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 1:   # single-channel 3-D
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)

            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError("Failed to encode numpy array to JPEG.")
            return buf.tobytes()

        # ---- raw bytes / bytearray -------------------------------------
        elif isinstance(image_input, (bytes, bytearray)):
            raw = bytes(image_input)

            # Try standard encoded formats first (JPEG, PNG, BMP, TIFF, WEBP…)
            np_arr = np.frombuffer(raw, dtype=np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                success, buf = cv2.imencode(".jpg", frame)
                if not success:
                    raise ValueError("Failed to re-encode image bytes to JPEG.")
                return buf.tobytes()

            # Fallback: raw numpy .tobytes() dump (uint8, 3-channel BGR)
            total = len(raw)
            if total % 3 == 0:
                pixel_count = total // 3
                for width in [3840, 2560, 1920, 1440, 1280, 1024, 960, 848, 800, 720, 640, 480, 320]:
                    if pixel_count % width == 0:
                        height = pixel_count // width
                        try:
                            frame   = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                            success, buf = cv2.imencode(".jpg", frame)
                            if success:
                                logging.warning(
                                    f"Decoded raw numpy bytes as {height}x{width} BGR frame."
                                )
                                return buf.tobytes()
                        except ValueError:
                            continue

            raise ValueError("Could not decode bytes as any known image format.")

        # ---- file path -------------------------------------------------
        elif isinstance(image_input, (str, Path)):
            frame = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError(f"Failed to encode '{image_input}' to JPEG.")
            return buf.tobytes()

        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def _append_to_json(self, record: dict):
        """Append one frame's detection record to the output JSON array file."""
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

    # ------------------------------------------------------------------
    # Main inference call
    # ------------------------------------------------------------------

    def text_prompt(self, image_input, prompts, conf=0.05):
        """
        Send a video frame to the YOLOE server and save the detection result.

        Parameters
        ----------
        image_input : np.ndarray | bytes | bytearray | str | Path
            The frame from the hailo (or any) pipeline.
            All formats are accepted — see _encode_frame() for details.

        prompts : list[str] | str
            Object class names to detect, e.g. ["Harness", "Hook"].

        conf : float
            Confidence threshold (default 0.05).

        Returns
        -------
        dict | None
            Parsed JSON response from the server, or None on failure.
            Response shape:
            {
                "inference_timestamp": "<ISO-8601>",
                "inference_start":     "<ISO-8601>",
                "detections": [
                    {
                        "prompt":       "Harness",
                        "confidence":   0.87,
                        "bounding_box": [x1, y1, x2, y2],
                        "polygon":      [[x, y], ...]
                    },
                    ...
                ]
            }
        """
        try:
            encoded_bytes = self._encode_frame(image_input)
        except (ValueError, TypeError) as e:
            logging.error(f"Frame encoding failed: {e}")
            return None

        prompts_str = (
            ",".join(str(p).strip() for p in prompts)
            if isinstance(prompts, list)
            else str(prompts)
        )

        files = {"file": ("frame.jpg", encoded_bytes, "image/jpeg")}
        data  = {"prompts": prompts_str, "conf": float(conf)}

        try:
            response = requests.post(
                YOLOE_SERVER_URL,
                files=files,
                data=data,
                timeout=120.0   # generous — first call may download mobileclip_blt.ts
            )
            response.raise_for_status()

            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError:
                logging.error(
                    f"Server returned non-JSON (status {response.status_code}). "
                    f"Body preview: '{response.text[:200]}'"
                )
                return None

        except requests.exceptions.Timeout:
            logging.error("YOLOE Server timed out — model may still be loading.")
            return None
        except requests.exceptions.ConnectionError:
            logging.error("Cannot reach YOLOE Server — is the Docker container running?")
            return None
        except requests.exceptions.HTTPError:
            logging.error(f"Server HTTP error {response.status_code}: {response.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None

        # Persist to disk
        self._append_to_json(result)

        # Forward to Kafka / downstream queue if one is attached
        if self.results_queue is not None:
            try:
                self.results_queue.put_nowait(result)
            except Exception:
                pass

        return result
