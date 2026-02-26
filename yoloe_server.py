from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO # Or YOLOE if you have a custom wrapper
import logging
import base64

logging.basicConfig(level=logging.INFO)

app = FastAPI()

MODEL_PATH = "/app/yoloe-11s-seg.pt" 
model = None

logging.info(f"Loading YOLOE model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH) 
    # Dummy run to initialize
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    logging.info("Model loaded and ready!")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    exit(1)

@app.post("/predict_prompt")
async def predict_prompt(
    file: UploadFile = File(...),
    prompts: str = Form(...),
    conf: float = Form(0.2)
):
    try:
        contents = await file.read()
        data = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Parse prompts and run inference
        prompt_list = [p.strip() for p in prompts.split(",")]
        model.set_classes(prompt_list)
        results = model.predict(source=frame, conf=conf, verbose=False)
        
        # Format the required JSON output
        detections = []
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls)
                    class_name = model.names[cls_id]
                    confidence = float(box.conf)
                    
                    # Get Bounding Box [x_min, y_min, x_max, y_max]
                    bbox = box.xyxy[0].tolist()
                    
                    # Get Polygon Segmentation Mask
                    polygon = []
                    if masks is not None and len(masks.xy) > i:
                        # masks.xy contains lists of [x, y] coordinates forming the polygon
                        polygon = masks.xy[i].tolist() 
                        
                    detections.append({
                        "prompt": class_name,
                        "bounding_box": bbox,
                        "polygon": polygon,
                        "confidence": confidence
                    })

        # Generate the annotated image with plotted boxes and masks
        annotated_frame = results[0].plot()
        success, encoded_image = cv2.imencode('.jpg', annotated_frame)
        
        base64_img = ""
        if success:
            # Convert the image to a base64 string so it can be sent inside JSON
            base64_img = base64.b64encode(encoded_image).decode('utf-8')

        # Return BOTH the detections and the encoded image
        return {
            "detections": detections, 
            "image_base64": base64_img
        }

    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
