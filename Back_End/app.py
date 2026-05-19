import os
import sys
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add the parent directory to the system path to import ml_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_core import DataPreprocessor, BreedClassifier, get_target_breeds, IMG_SIZE, WEIGHTS_PATH, DATASET_DIR

app = FastAPI(title="Cattle Breed Classifier API")

# Enable CORS so the React frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration paths relative to the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(BASE_DIR, DATASET_DIR)
weights_path = os.path.join(BASE_DIR, WEIGHTS_PATH)

TARGET_BREEDS = get_target_breeds(dataset_path)

if not TARGET_BREEDS:
    print(f"Warning: Directory {dataset_path} not found or empty. Make sure dataset is available.")

# Initialize model on startup
try:
    preprocessor = DataPreprocessor(target_size=IMG_SIZE)
    classifier = BreedClassifier(num_classes=len(TARGET_BREEDS) if TARGET_BREEDS else 1, model_weights_path=weights_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    preprocessor, classifier = None, None

@app.post("/predict")
async def predict_breeds(files: list[UploadFile] = File(...)):
    if not classifier or not TARGET_BREEDS:
        return {"error": "Model or dataset not properly loaded."}

    results = []
    
    for file in files:
        # 1. Read file into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            results.append({"filename": file.filename, "error": "Invalid image format"})
            continue
            
        # 2. Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Preprocess and Predict
        processed_tensor = preprocessor.preprocess(image)
        breed, conf = classifier.predict(processed_tensor, TARGET_BREEDS)
        
        results.append({
            "filename": file.filename,
            "breed": breed,
            "confidence": f"{conf}%"
        })
        
    return {"results": results}

if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)