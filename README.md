# Cattle Breed Classification

An end-to-end AI application that leverages Transfer Learning (MobileNetV2) to classify cattle breeds from images. The project features a FastAPI backend for high-performance model serving, a React frontend for easy user interaction, and a centralized machine learning core for easy maintenance and training.

## Features

- **Centralized ML Core:** All Machine Learning logic (model building, preprocessing, prediction) is centralized in `ml_core.py` following DRY (Don't Repeat Yourself) principles.
- **Deep Learning Engine:** Utilizes a pre-trained MobileNetV2 model, fine-tuned for livestock breed classification.
- **FastAPI Backend:** A robust, high-speed REST API that handles multi-image uploads directly in memory, separate from ML logic.
- **React UI:** A modern frontend application allowing users to upload single or multiple images and view breed predictions alongside confidence scores.
- **CLI Testing Tools:** Standalone Python scripts to test the inference model via the terminal and train the model locally.

## Project Structure

```text
.
├── cattle/                            # Dataset directory (e.g., cattle/Amritmahal/, cattle/Gir/)
├── ml_core.py                         # Centralized Machine Learning Core (Model, Preprocessor, classes)
├── Back_End/                          
│   └── app.py                         # FastAPI backend server
├── Front_End/                         # React frontend application
├── Testing_model.py                   # CLI script for local one-shot testing inference
├── Training_model.py                  # Script for model training locally
└── livestock_mobilenetv2.weights.h5   # Saved model weights (generated after initial training)
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

Ensure your directory structure looks like this so the backend can dynamically read the breed names:

```text
cattle/
  ├── Amritmahal/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── Gir/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── ...
```

> **Note:** The backend dynamically reads these folder names to register the target breeds.

### 2. Model Training (Optional)

If you need to train the model from scratch, simply run the training script:

```bash
python Training_model.py
```
This will generate the `livestock_mobilenetv2.weights.h5` file, which contains the trained weights used for classification.

### 3. Backend Setup (FastAPI)

Install the required Python dependencies:

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy tensorflow
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

It will prompt you to enter the path of an image, print the detected breed and confidence, and exit automatically.

## Technologies Used

* **Machine Learning:** TensorFlow, Keras (MobileNetV2), OpenCV, NumPy
* **Backend:** FastAPI, Uvicorn, Python
* **Frontend:** React, JavaScript, CSS
