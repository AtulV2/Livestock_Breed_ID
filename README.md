# Cattle Breed Classification

An end-to-end AI application that leverages a two-step pipeline for robust cattle breed identification. First, a YOLOv8 object detection model locates and crops the cattle in the image. Then, a fine-tuned MobileNetV2 model classifies the specific breed. The project features a FastAPI backend for high-performance model serving, a React frontend for easy user interaction, and a centralized machine learning core.

## Features

- **Two-Step ML Pipeline:** 
  1. **Detection:** Ultralytics YOLOv8s detects and crops the cattle from the background, ensuring high accuracy even in cluttered environments.
  2. **Classification:** A pre-trained MobileNetV2 model fine-tuned for livestock breed classification determines the specific breed.
- **Centralized ML Core:** All Machine Learning logic (model building, preprocessing, prediction, detection) is centralized in `ml_core.py`.
- **FastAPI Backend:** A robust, high-speed REST API that handles multi-image uploads directly in memory, integrating both the YOLO and MobileNet models.
- **React UI:** A modern frontend application allowing users to upload single or multiple images and view breed predictions alongside confidence scores, as well as handling "No Cattle Detected" scenarios.
- **CLI Testing Tools:** Standalone Python scripts to test the inference model via the terminal and train the classification model locally.

## Project Structure

```text
.
├── cattle/                            # Dataset directory (e.g., cattle/Amritmahal/, cattle/Gir/)
├── ml_core.py                         # Centralized Machine Learning Core (YOLO, MobileNetV2)
├── Back_End/                          
│   └── app.py                         # FastAPI backend server
├── Front_End/                         # React frontend application
├── Testing_model.py                   # CLI script for local one-shot testing inference
├── Training_model.py                  # Script for model training locally
├── yolov8s.pt                         # YOLOv8s weights for cattle detection
└── livestock_mobilenetv2.weights.h5   # MobileNetV2 saved model weights
```

## Prerequisites

* **Python 3.8+**
* **Node.js 18+** (for the React frontend)

## 🚀 Setup & Installation

### 1. Dataset Preparation

This project uses the **Indian Cattle Image Dataset** from Kaggle. 

1. Download the dataset from here: [Indian Cattle Image Dataset](https://www.kaggle.com/datasets/atharvadarpude/indian-cattle-image-dataset)
2. Extract the downloaded `.zip` file.
3. Rename the extracted main folder to `cattle` and place it directly in the root directory of this project.

### 2. Model Training (Optional)

If you need to train the classification model from scratch, simply run the training script:

```bash
python Training_model.py
```
This will generate the `livestock_mobilenetv2.weights.h5` file. Note that `yolov8s.pt` will be downloaded automatically by the `ultralytics` package if not present.

### 3. Backend Setup (FastAPI)

Install the required Python dependencies:

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy tensorflow ultralytics
```

Start the FastAPI server:

```bash
python Back_End/app.py
```

The API will be running at `http://localhost:8000`.

### 4. Frontend Setup (React)

Open a new terminal window and navigate to the frontend directory:

```bash
cd Front_End
npm install
npm run dev
```

### 5. CLI Testing

If you want to quickly test an image without starting the web servers, use the standalone script:

```bash
python Testing_model.py
```

## Technologies Used

* **Object Detection:** Ultralytics YOLOv8
* **Image Classification:** TensorFlow, Keras (MobileNetV2)
* **Computer Vision:** OpenCV, NumPy
* **Backend:** FastAPI, Uvicorn, Python
* **Frontend:** React, JavaScript, CSS, Vite
