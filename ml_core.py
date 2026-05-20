import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Configuration constants
DATASET_DIR = 'cattle'
WEIGHTS_PATH = 'livestock_mobilenetv2.weights.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_target_breeds(dataset_dir=DATASET_DIR):
    """Dynamically get class names based on folder structure."""
    if os.path.exists(dataset_dir):
        return sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    return []

class CattleDetector:
    """Step 1: Detects ONLY cattle using YOLOv8 with high sensitivity."""
    def __init__(self, model_name='yolov8s.pt'): 
        # Using yolov8s.pt for better accuracy than the 'nano' version
        self.model = YOLO(model_name)
        # COCO ID 19 is strictly for 'cow'
        self.target_class = 19 

    def detect_and_crop(self, image):
        """Returns a cropped image of the cow if found, otherwise None."""
        # conf=0.20 makes the detector very sensitive to cows in the background
        results = self.model(image, conf=0.20, verbose=False) 
        
        for result in results:
            for box in result.boxes:
                # Check ONLY for the cow class
                if int(box.cls[0]) == self.target_class:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Add 10% padding to ensure the whole animal is visible for the classifier
                    h, w, _ = image.shape
                    pad_w = int((x2 - x1) * 0.1)
                    pad_h = int((y2 - y1) * 0.1)
                    
                    x1_new = max(0, x1 - pad_w)
                    y1_new = max(0, y1 - pad_h)
                    x2_new = min(w, x2 + pad_w)
                    y2_new = min(h, y2 + pad_h)
                    
                    return image[y1_new:y2_new, x1_new:x2_new]
        return None

class ImageHandler:
    """Manages I/O operations for cattle images."""
    @staticmethod
    def load_image(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found at path: {filepath}")
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class DataPreprocessor:
    """Handles the conversion of raw images into model format."""
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size

    def preprocess(self, image):
        resized_image = cv2.resize(image, self.target_size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        return np.expand_dims(normalized_image, axis=0)

class BreedClassifier:
    """Step 2: The engine for classifying the specific cattle breed."""
    def __init__(self, num_classes, model_weights_path=WEIGHTS_PATH):
        self.num_classes = num_classes
        self.model = self._build_model()
        if model_weights_path and os.path.exists(model_weights_path):
            self.load_model(model_weights_path)

    def _build_model(self):
        """Constructs and compiles the MobileNetV2 architecture."""
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        base_model.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, processed_image, class_names):
        """Performs breed classification on the cropped image."""
        predictions = self.model.predict(processed_image, verbose=0)
        idx = np.argmax(predictions[0])
        conf = np.max(predictions[0])
        return class_names[idx], round(float(conf) * 100, 2)