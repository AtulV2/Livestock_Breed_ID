import os
import sys
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add the parent directory to the system path to import ml_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_core import (
    DataPreprocessor, 
    BreedClassifier, 
    CattleDetector, 
    get_target_breeds, 
    IMG_SIZE, 
    WEIGHTS_PATH, 
    DATASET_DIR
)

app = FastAPI(title="Cattle Breed Classifier API with Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(BASE_DIR, DATASET_DIR)
weights_path = os.path.join(BASE_DIR, WEIGHTS_PATH)

TARGET_BREEDS = get_target_breeds(dataset_path)

# Initialize models
try:
    detector = CattleDetector() # Step 1: YOLO Cow Detector
    preprocessor = DataPreprocessor(target_size=IMG_SIZE)
    classifier = BreedClassifier(num_classes=len(TARGET_BREEDS) if TARGET_BREEDS else 1, model_weights_path=weights_path)
except Exception as e:
    print(f"Failed to load models: {e}")
    detector, preprocessor, classifier = None, None, None
    
@app.post("/predict")
async def predict_breeds(files: list[UploadFile] = File(...)):
    if not classifier or not detector:
        return {"error": "Models not properly loaded on server."}

    results = []
    
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            results.append({"filename": file.filename, "error": "Invalid image format"})
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Detection Step
        cropped_cow = detector.detect_and_crop(image_rgb)
        
        if cropped_cow is None:
            # CHANGE: Set breed to specific message and confidence to None
            results.append({
                "filename": file.filename,
                "breed": "No Cattle Detected",
                "confidence": None, # UI will check for this
                "detected": False
            })
            continue

        # 2. Classification Step
        processed_tensor = preprocessor.preprocess(cropped_cow)
        breed, conf = classifier.predict(processed_tensor, TARGET_BREEDS)
        
        results.append({
            "filename": file.filename,
            "breed": breed,
            "confidence": f"{conf}%",
            "detected": True
        })
        
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)