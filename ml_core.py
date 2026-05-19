import os
import cv2
import numpy as np
import tensorflow as tf

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

class ImageHandler:
    """Manages I/O operations for cattle images."""
    @staticmethod
    def load_image(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found at path: {filepath}")
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image. Ensure it is a valid format: {filepath}")
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
    """The core engine encapsulating the deep learning model."""
    def __init__(self, num_classes, model_weights_path=WEIGHTS_PATH):
        self.num_classes = num_classes
        self.model = self._build_model()
        
        if model_weights_path and os.path.exists(model_weights_path):
            self.load_model(model_weights_path)
        else:
            print("No pre-trained weights found. The model might need to be trained.")

    def _build_model(self):
        """Constructs and compiles the MobileNetV2 Transfer Learning Architecture."""
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        base_model.trainable = False # Freeze base layers
        
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train_model(self, train_dataset, val_dataset, epochs=10, save_path=WEIGHTS_PATH):
        """Trains the model on the provided datasets and saves the weights."""
        print("\nStarting model training...")
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stop]
        )
        
        self.model.save_weights(save_path)
        print(f"\nTraining complete. Weights saved to {save_path}")
        return history

    def load_model(self, weights_path):
        """Loads trained weights into the model architecture."""
        print(f"Loading weights from {weights_path}...")
        self.model.load_weights(weights_path)
        print("Weights loaded successfully.")

    def predict(self, processed_image, class_names):
        """Performs the forward pass to determine breed category."""
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return class_names[predicted_class_index], round(float(confidence) * 100, 2)
