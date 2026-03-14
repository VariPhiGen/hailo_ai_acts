import os
import json
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import requests


# URL for the external YOLOE inference service.
# Override with the YOLOE_API_URL environment variable.
YOLOE_API_URL: str = os.getenv(
    "YOLOE_API_URL",
    "http://yoloe.vgiskill.com/predict_prompt",
)

# Minimum confidence threshold sent to the server.
# The server accepts a float 0–1; the API default is 0.1.
YOLOE_CONF: float = float(os.getenv("YOLOE_CONF", "0.1"))


def _encode_image_to_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """
    Encode a NumPy image (BGR or RGB) to JPEG bytes.

    Args:
        image: NumPy array image (H x W x 3).
        quality: JPEG quality (0–100).

    Returns:
        Encoded JPEG bytes.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid NumPy ndarray")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected image with shape (H, W, 3), got {image.shape}")

    # OpenCV expects BGR; if image is float, convert to uint8 safely
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    success, buffer = cv2.imencode(".jpg", img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("failed to JPEG-encode image")

    return buffer.tobytes()


def text_prompt(
    image: np.ndarray,
    prompt: List[str],
    *,
    timeout: float = 15.0,
    api_url: Optional[str] = None,
    conf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call the YOLOE inference service with a NumPy image and a list of class prompts.

    API contract (POST http://yoloe.vgiskill.com/predict_prompt):
        Request  – multipart/form-data:
            file    : JPEG image bytes  (field name: "file")
            prompts : comma-separated class names  (e.g. "scaffolding, safety_harness")
            conf    : float confidence threshold (optional, default 0.1)

        Response JSON:
            {
              "inference_timestamp": str,
              "inference_start":     str,
              "detections": [
                {
                  "prompt":       str,
                  "confidence":   float,
                  "bounding_box": [x1, y1, x2, y2],  # absolute pixel coords
                  "polygon":      [[x, y], ...]
                },
                ...
              ]
            }

    Args:
        image:   Input image as NumPy array (H x W x 3, BGR or RGB).
                 MUST be in original (full-resolution) pixel space so that the
                 returned coordinates align with Hailo detection_boxes.
        prompt:  List of class names to detect (e.g. ["scaffolding", "harness"]).
        timeout: HTTP request timeout in seconds.
        api_url: Optional override for the API URL. Defaults to YOLOE_API_URL env var.
        conf:    Optional confidence threshold override. Defaults to YOLOE_CONF env var.

    Returns:
        Dict with the following keys, each a list (one entry per detection):
            "prompt"       – list[str]              detected class names
            "bounding_box" – list[dict]             {"x1", "y1", "x2", "y2"}
            "polygon"      – list[list[list[float]]]  [[x,y], ...]
            "confidence"   – list[float]
    """
    if not prompt:
        raise ValueError("prompt list must not be empty")

    url = api_url or YOLOE_API_URL
    threshold = conf if conf is not None else YOLOE_CONF

    # Encode image to JPEG bytes
    image_bytes = _encode_image_to_jpeg(image)

    # Build multipart/form-data payload matching the new API contract:
    #   file    → JPEG bytes
    #   prompts → comma-separated string  (NOT JSON)
    #   conf    → float as string
    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg"),
    }
    data = {
        "prompts": ", ".join(prompt),
        "conf": str(threshold),
    }

    _debug = os.environ.get("DEBUG_MODE") == "1"

    try:
        if _debug:
            print(f"🔍 [YOLOE] POST {url} | prompts={', '.join(prompt)} | conf={threshold} | image={image.shape}")
        response = requests.post(url, files=files, data=data, timeout=timeout)
        response.raise_for_status()
        if _debug:
            print(f"✅ [YOLOE] HTTP {response.status_code} in {response.elapsed.total_seconds():.2f}s")
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call YOLOE API at {url}: {exc}") from exc

    payload = response.json()

    # Expect top-level "detections" list; raise clearly if missing
    if "detections" not in payload:
        error_msg = payload.get("error", payload.get("detail", "No 'detections' key in response"))
        raise RuntimeError(f"YOLOE API unexpected response: {error_msg}")

    prompts_out: List[str] = []
    bboxes_out: List[Dict[str, float]] = []
    polygons_out: List[List[List[float]]] = []
    confidences_out: List[float] = []

    for det in payload.get("detections", []):
        # Class name
        prompts_out.append(det.get("prompt", ""))

        # Bounding box: API returns [x1, y1, x2, y2] list → normalise to dict
        raw_bbox = det.get("bounding_box", [])
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bboxes_out.append({
                "x1": float(raw_bbox[0]),
                "y1": float(raw_bbox[1]),
                "x2": float(raw_bbox[2]),
                "y2": float(raw_bbox[3]),
            })
        else:
            bboxes_out.append({})

        # Polygon: [[x,y], ...] list of 2-element lists
        raw_poly = det.get("polygon") or []
        polygons_out.append([[float(pt[0]), float(pt[1])] for pt in raw_poly if len(pt) == 2])

        # Confidence
        confidences_out.append(float(det.get("confidence", 0.0)))

    return {
        "prompt": prompts_out,
        "bounding_box": bboxes_out,
        "polygon": polygons_out,
        "confidence": confidences_out,
    }
