from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from ultralytics import YOLO
import cv2
import base64

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("../runs/detect/train/weights/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image prediction requests"""
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid image file"}
            )

        # Make prediction
        results = model.predict(img, conf=0.25)
        result = results[0]
        
        # Process detection
        has_pneumonia = len(result.boxes) > 0
        confidence = 0
        
        # Draw boxes if pneumonia detected
        if has_pneumonia:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(img, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Add confidence text
                cv2.putText(img, 
                           f'Pneumonia: {confidence:.2f}', 
                           (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)

        # Convert image to base64
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        response = JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "has_pneumonia": has_pneumonia,
                "confidence": confidence,
                "image_data": f"data:image/jpeg;base64,{img_base64}"
            }
        )
        
        # Add no-cache headers
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}