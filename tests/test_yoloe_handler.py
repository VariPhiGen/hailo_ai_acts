"""
Unit tests for basic_pipelines/yoloe_handler.py

Tests the new external API contract:
  POST http://yoloe.vgiskill.com/predict_prompt
  Form-data: file (JPEG), prompts (comma-separated str), conf (float)
  Response:  {"detections": [{"prompt", "confidence", "bounding_box", "polygon"}]}
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure the basic_pipelines package is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basic_pipelines.yoloe_handler import text_prompt, _encode_image_to_jpeg


def _make_fake_image(h=480, w=640) -> np.ndarray:
    """Return a solid-colour BGR uint8 image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_api_response(detections):
    """Build a minimal API-compatible response dict."""
    return {
        "inference_timestamp": "2026-03-13T14:00:00+00:00",
        "inference_start": "2026-03-13T13:59:59+00:00",
        "detections": detections,
    }


class TestEncodeImage(unittest.TestCase):

    def test_encode_valid_image(self):
        img = _make_fake_image()
        result = _encode_image_to_jpeg(img)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    def test_encode_rejects_none(self):
        with self.assertRaises(ValueError):
            _encode_image_to_jpeg(None)

    def test_encode_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            _encode_image_to_jpeg(np.zeros((10, 10), dtype=np.uint8))

    def test_encode_float32_image(self):
        img = np.ones((100, 100, 3), dtype=np.float32) * 128
        result = _encode_image_to_jpeg(img)
        self.assertIsInstance(result, bytes)


class TestTextPromptNewAPI(unittest.TestCase):

    def _mock_response(self, detections):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_api_response(detections)
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_single_detection_parsed_correctly(self, mock_post):
        """Full detection parsed into expected dict shape."""
        mock_post.return_value = self._mock_response([
            {
                "prompt": "scaffolding",
                "confidence": 0.651,
                "bounding_box": [139.9, 43.2, 497.4, 636.5],
                "polygon": [[297.0, 181.0], [418.0, 137.0], [350.0, 200.0]],
            }
        ])

        result = text_prompt(_make_fake_image(), ["scaffolding"])

        self.assertEqual(result["prompt"], ["scaffolding"])
        self.assertAlmostEqual(result["confidence"][0], 0.651, places=2)
        bbox = result["bounding_box"][0]
        self.assertAlmostEqual(bbox["x1"], 139.9, places=1)
        self.assertAlmostEqual(bbox["y1"], 43.2, places=1)
        self.assertAlmostEqual(bbox["x2"], 497.4, places=1)
        self.assertAlmostEqual(bbox["y2"], 636.5, places=1)
        self.assertEqual(len(result["polygon"][0]), 3)

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_multi_class_prompts_joined_correctly(self, mock_post):
        """Multiple prompts are joined as comma-separated string in the request."""
        mock_post.return_value = self._mock_response([])

        text_prompt(_make_fake_image(), ["scaffolding", "safety_harness"])

        _, kwargs = mock_post.call_args
        sent_data = kwargs.get("data", {})
        self.assertEqual(sent_data["prompts"], "scaffolding, safety_harness")
        self.assertIn("conf", sent_data)  # confidence threshold always sent

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_conf_field_sent(self, mock_post):
        """Default YOLOE_CONF (0.1) is always sent in the request."""
        mock_post.return_value = self._mock_response([])
        text_prompt(_make_fake_image(), ["scaffolding"])
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["data"]["conf"], "0.1")

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_conf_override(self, mock_post):
        """Per-call conf override is honoured."""
        mock_post.return_value = self._mock_response([])
        text_prompt(_make_fake_image(), ["scaffolding"], conf=0.5)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["data"]["conf"], "0.5")

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_empty_detections_returns_four_empty_lists(self, mock_post):
        """When server returns no detections, all four lists are empty."""
        mock_post.return_value = self._mock_response([])
        result = text_prompt(_make_fake_image(), ["scaffolding"])
        self.assertEqual(result["prompt"], [])
        self.assertEqual(result["bounding_box"], [])
        self.assertEqual(result["polygon"], [])
        self.assertEqual(result["confidence"], [])

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_missing_detections_key_raises(self, mock_post):
        """Response without 'detections' key raises RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "model not loaded"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError, msg="Should raise on missing 'detections'"):
            text_prompt(_make_fake_image(), ["scaffolding"])

    @patch("basic_pipelines.yoloe_handler.requests.post")
    def test_missing_polygon_handled_gracefully(self, mock_post):
        """Detection with null/missing polygon returns empty polygon list."""
        mock_post.return_value = self._mock_response([
            {
                "prompt": "scaffolding",
                "confidence": 0.5,
                "bounding_box": [0, 0, 100, 100],
                "polygon": None,  # server may return null
            }
        ])
        result = text_prompt(_make_fake_image(), ["scaffolding"])
        self.assertEqual(result["polygon"], [[]])

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            text_prompt(_make_fake_image(), [])


if __name__ == "__main__":
    unittest.main()
