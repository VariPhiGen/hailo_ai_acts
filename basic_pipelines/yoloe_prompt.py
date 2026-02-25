import cv2
import numpy as np
import requests
import json
import argparse
import logging
import base64
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

YOLOE_SERVER_URL = "http://localhost:5000/predict_prompt"

class YOLOEHandler:
    def __init__(self, config=None):
        """
        config: The full configuration.json dictionary (used during integration).
                If None, the handler is running in standalone CLI mode.
        """
        self.config = config or {}
        self.results_queue = None

    def set_results_queue(self, queue):
        """Used later when integrating with detection.py to pass results to Kafka"""
        self.results_queue = queue

    def text_prompt(self, image_input, prompts, conf=0.05):
        # ... (Keep steps 1 to 4 exactly the same: Handle input, encode, prepare request) ...
        # (Assuming you haven't changed the top part of the function)
        
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None: return None
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else: return None

        prompts_str = ",".join([str(p).strip() for p in prompts]) if isinstance(prompts, list) else str(prompts)
        success, img_encoded = cv2.imencode('.jpg', frame)
        if not success: return None

        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'prompts': prompts_str, 'conf': float(conf)}

        # --- 5. Send Request and Process Response ---
        try:
            response = requests.post(YOLOE_SERVER_URL, files=files, data=data, timeout=15.0)

            if response.status_code == 200:
                result_json = response.json()
                
                # Pop the image data out of the JSON so it doesn't clutter the raw data
                image_b64 = result_json.pop("image_base64", None)
                
                # --- VISUALIZATION & SAVING (ONLY FOR CLI / TEST MODE) ---
                if isinstance(image_input, str):
                    # 1. Save JSON to file
                    output_json_path = "yoloe_test_output.json"
                    with open(output_json_path, "w") as f:
                        json.dump(result_json, f, indent=4)
                    print(f"\n✅ Saved full JSON results to '{output_json_path}'")
                    
                    # 2. Decode and display the image
                    if image_b64:
                        img_data = base64.b64decode(image_b64)
                        np_arr = np.frombuffer(img_data, np.uint8)
                        img_to_show = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        print("🖼️ Displaying annotated image. Press any key in the window to close...")
                        cv2.imshow("YOLOE Prompt-Based Detection", img_to_show)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                
                return result_json
            else:
                logging.error(f"YOLOE Server returned an error {response.status_code}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Connection to YOLOE Server failed: {e}")
            return None


# =====================================================================
# CLI EXECUTION BLOCK FOR STANDALONE TESTING
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOE Prompt-based Detection Handler")
    parser.add_argument("--image", type=str, required=True, help="Path to the local image file (jpg, png, etc.)")
    parser.add_argument("--prompts", type=str, required=True, help="Comma-separated list of items to detect (e.g., 'Harness, Hook')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    # Initialize handler without the complex JSON config for simple testing
    handler = YOLOEHandler()
    
    # Run the prompt detection function
    logging.info(f"Sending request to YOLOE Server...")
    handler.text_prompt(image_input=args.image, prompts=args.prompts, conf=args.conf)
