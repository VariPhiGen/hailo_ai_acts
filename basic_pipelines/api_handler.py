import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from form_util import fill_form
from helper_utils import make_labelled_image


class APIHandler:
    """Handles API-related operations including S3 uploads and form submissions."""

    def __init__(self, executor: ThreadPoolExecutor, 
                 upload_to_s3_safe: Callable,
                 get_api_s3_url: Callable,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize API Handler.
        
        Args:
            executor: ThreadPoolExecutor for parallel uploads
            upload_to_s3_safe: Function to upload files to S3 (signature: (bytes, str, bool) -> Optional[str])
            get_api_s3_url: Function to get API S3 URL (signature: (str, str) -> Optional[str])
            config: Optional configuration dictionary
        """
        self.executor = executor
        self.upload_to_s3_safe = upload_to_s3_safe
        self.get_api_s3_url = get_api_s3_url
        self.config = config or {}

    def _extract_category_subcategory(self, message: Dict[str, Any]) -> Tuple[str, str]:
        """Extract event category and subcategory from message."""
        try:
            subcategory_full = message["absolute_bbox"][0]["subcategory"]
            parts = subcategory_full.split("-", 1)
            event_category = parts[0] if len(parts) > 0 else subcategory_full
            event_subcategory = parts[1] if len(parts) > 1 else ""
            return event_category, event_subcategory
        except (KeyError, IndexError, AttributeError) as e:
            print(f"DEBUG: Error extracting category/subcategory: {e}")
            return "Unknown", ""

    def _build_form_data(self, message: Dict[str, Any], 
                        image_s3_url: str, 
                        labelled_image_s3_url: str, 
                        video_s3_url: str,
                        event_category: str,
                        event_subcategory: str) -> Dict[str, Any]:
        """Build form data payload for API submission."""
        return {
            "id": "7f148215-d1c5-4b83-bf58-3a2249d4b107",
            "timestamp": message.get("datetimestamp", ""),
            "image_original": image_s3_url,
            "image_labelled": labelled_image_s3_url,
            "video_clip": video_s3_url,
            "remarks": "Violation Detected",
            "event_category": event_category,
            "whom_to_notify": "mohammed@arresto.in",
            "camera_unique_id": "3acbb55c-d6cb-4470-8d9c-5d8de5223bea",
            "safety_captain_id": "varun@arresto.in",
            "incident_occured_on": message.get("datetimestamp_trackerid", ""),
            "incident_updated_on": message.get("datetimestamp_trackerid", ""),
            "action_status": "Pending",
            "event_subcategory": event_subcategory,
            "sverity":"medium",
            "count":1
        }

    def process_and_submit(self, message: Dict[str, Any], topic: str = "") -> bool:
        """
        Process message, upload files to API S3, and submit form data.
        
        Args:
            message: Message dictionary containing image_bytes, snap_shot_bytes, video_bytes, etc.
            topic: Optional topic name for logging
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get file bytes from message
            image_bytes = message.get("org_img")
            snap_shot_bytes = message.get("snap_shot")
            video_bytes = message.get("video")
            # Extract category and subcategory
            event_category, event_subcategory = self._extract_category_subcategory(message)

            if not image_bytes:
                print("DEBUG: No image bytes in message, skipping API submission")
                return False

            # Create labelled image first (needed before parallel uploads)
            labelled_image_bytes = make_labelled_image(message,event_category,event_subcategory)

            # Parallel uploads to API S3
            api_uploads = {}
            api_futures = {
                self.executor.submit(self.upload_to_s3_safe, image_bytes, "image", True): "org_img",
                self.executor.submit(self.upload_to_s3_safe, labelled_image_bytes, "image", True): "labelled_img",
                self.executor.submit(self.upload_to_s3_safe, snap_shot_bytes, "snapshot", True): "snap_shot",
                self.executor.submit(self.upload_to_s3_safe, video_bytes, "video", True): "video"
            }

            for fut in as_completed(api_futures):
                key = api_futures[fut]
                try:
                    api_uploads[key] = fut.result()
                except Exception as e:
                    print(f"DEBUG: API S3 upload task failed for {key}: {e}")
                    api_uploads[key] = None

            # Check if labelled image upload succeeded
            if not api_uploads.get("labelled_img"):
                print("DEBUG: Labelled image upload failed, skipping API submission")
                return False

            # Get uploaded filenames
            image_filename = api_uploads.get("org_img")
            labelled_image_filename = api_uploads.get("labelled_img")
            video_filename = api_uploads.get("video")

            # Construct full URLs from API S3 configs
            image_s3_url = self.get_api_s3_url(image_filename, "image") if image_filename else ""
            labelled_image_s3_url = self.get_api_s3_url(labelled_image_filename, "image") if labelled_image_filename else ""
            video_s3_url = self.get_api_s3_url(video_filename, "video") if video_filename else ""

            # Update message with filename
            message["org_img"] = image_filename

            # Build form data
            form_data = self._build_form_data(
                message, image_s3_url, labelled_image_s3_url, video_s3_url,
                event_category, event_subcategory
            )

            # Resolve template path (config override or default next to this file)
            template_path = Path(
                self.config.get(
                    "api_template_path",
                    Path(__file__).resolve().parent / "template.json",
                )
            )
            if not template_path.is_file():
                print(f"DEBUG: API template not found at {template_path}")
                return False

            # Submit form
            post_url = self.config.get("api_post_url") or os.getenv("API_POST_URL")
            response = fill_form(str(template_path), form_data, post_url=post_url)
            #print(f"Successfully sent message to {topic}, {response}")
            return True

        except Exception as e:
            print(f"DEBUG: API processing failed: {e}")
            return False

