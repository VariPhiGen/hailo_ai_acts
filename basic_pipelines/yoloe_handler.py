import os
import json
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import requests


# Base URL for the YOLOE/YOLOv26 segmentation API.
# By default, expects the FastAPI service from `app.py` to be running on localhost:8000.
# You can override this with the `YOLOE_API_URL` environment variable.
YOLOE_API_URL: str = os.getenv("YOLOE_API_URL", "http://127.0.0.1:8000/predict")


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
) -> Dict[str, Any]:
    """
    Call the YOLOE/YOLOv26 segmentation API with a NumPy image and a list of prompts.

    Args:
        image: Input image as NumPy array (H x W x 3, BGR or RGB).
        prompt: List of class names / text prompts.
        timeout: HTTP request timeout in seconds.
        api_url: Optional override for the API URL. Defaults to `YOLOE_API_URL`.

    Returns:
        JSON-like dict with the following keys, each containing a list:
            - "prompt":       list of detected class names (str)
            - "bounding_box": list of bbox dicts: {"x1", "y1", "x2", "y2"}
            - "polygon":      list of segmentation polygons (list[list[float]])
            - "confidence":   list of confidences (float)
    """
    if not prompt:
        raise ValueError("prompt list must not be empty")

    url = api_url or YOLOE_API_URL

    # Prepare multipart/form-data payload:
    #   - image: JPEG-encoded bytes
    #   - classes: JSON string of prompt list (matches FastAPI /predict contract)
    image_bytes = _encode_image_to_jpeg(image)

    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg"),
    }
    data = {
        "classes": json.dumps(prompt),
    }

    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call YOLOE API at {url}: {exc}") from exc

    payload = response.json()

    # The segmentation API (see `app.py` /predict) returns:
    # {
    #   "success": bool,
    #   "results": [
    #       {
    #           "class_name": str,
    #           "confidence": float,
    #           "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
    #           "segmentation_mask_polygon": [[x, y], ...] | null,
    #           ...
    #       },
    #       ...
    #   ],
    #   "image_shape": {"width": int, "height": int},
    #   "inference_time_ms": float | null,
    #   "error": str | null
    # }

    if not payload.get("success", False):
        error_msg = payload.get("error", "Unknown error from YOLOE API")
        raise RuntimeError(f"YOLOE API returned error: {error_msg}")

    prompts_out: List[str] = []
    bboxes_out: List[Dict[str, float]] = []
    polygons_out: List[List[List[float]]] = []
    confidences_out: List[float] = []

    for det in payload.get("results", []):
        prompts_out.append(det.get("class_name", ""))
        bboxes_out.append(det.get("bbox", {}))
        polygons_out.append(det.get("segmentation_mask_polygon") or [])
        confidences_out.append(det.get("confidence", 0.0))

    return {
        "prompt": prompts_out,
        "bounding_box": bboxes_out,
        "polygon": polygons_out,
        "confidence": confidences_out,
    }
