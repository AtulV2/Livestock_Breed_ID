import os
import tensorflow as tf
from datetime import datetime
from ml_core import ImageHandler, DataPreprocessor, BreedClassifier, get_target_breeds, DATASET_DIR, WEIGHTS_PATH, BATCH_SIZE, IMG_SIZE

class CensusRecord:
    """Data Transfer Object (DTO) structuring results for the digital census DB."""
    
    def __init__(self, breed_name, confidence_level):
        self.breed_name = breed_name
        self.confidence_level = round(float(confidence_level) * 100, 2)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_to_db(self):
        """Mocks the database injection process."""
        db_payload = {
            "breed": self.breed_name,
            "confidence": self.confidence_level,
            "recorded_at": self.timestamp
        }
        success = True # Simulating a successful DB transaction
        
        if success:
            return "Record successfully pushed to Digital Census DB."
        else:
            return "Failed to push record to database."

# ==========================================
# Implementation Logic (Main Execution)
# ==========================================
if __name__ == "__main__":
    print("--- Livestock Breed ID System Initializing ---")
    
    # Pre-check if directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"CRITICAL ERROR: Directory '{DATASET_DIR}' not found.")
        print("Please ensure your dataset folder is named 'cattle' and is in the same directory as this script.")
        exit()

    # Load Dataset directly from the folder structure
    print("Loading dataset...")
    
    # Check total files
    total_files = sum([len(files) for r, d, files in os.walk(DATASET_DIR) if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files)])
    
    if total_files > 5:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, validation_split=0.2, subset="training", seed=123,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
        val_ds = train_ds # Use training set as validation for tiny demo datasets

    TARGET_BREEDS = train_ds.class_names 
    print(f"Detected Breeds: {TARGET_BREEDS}")
    
    # Normalize the datasets (scale pixels to 0-1) for training
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Initialize the Classifier
    classifier = BreedClassifier(num_classes=len(TARGET_BREEDS), model_weights_path=WEIGHTS_PATH)
    
    # Train the model if weights don't exist
    if not os.path.exists(WEIGHTS_PATH):
        print("\nInitiating Training Phase...")
        classifier.train_model(train_ds, val_ds, epochs=10, save_path=WEIGHTS_PATH)
    else:
        print("\nPre-trained weights found. Skipping training phase.")

    # Inference Testing (Simulating a Field Officer's capture)
    print("\n--- Testing Inference ---")
    image_handler = ImageHandler()
    preprocessor = DataPreprocessor(target_size=IMG_SIZE)
    
    # Safely grab the first valid image from the first breed folder for testing
    first_breed_dir = os.path.join(DATASET_DIR, TARGET_BREEDS[0])
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    sample_images = []
    if os.path.exists(first_breed_dir):
        sample_images = [f for f in os.listdir(first_breed_dir) if f.lower().endswith(valid_extensions)]
    
    if sample_images:
        test_image_path = os.path.join(first_breed_dir, sample_images[0])
        
        try:
            raw_image = image_handler.load_image(test_image_path)
            processed_tensor = preprocessor.preprocess(raw_image)
            
            breed, conf = classifier.predict(processed_tensor, class_names=TARGET_BREEDS)
            
            record = CensusRecord(breed_name=breed, confidence_level=conf)
            db_status = record.save_to_db()
            
            print(f"Tested Image: {test_image_path}")
            print("\n--- Output ---")
            print(f"o Detected Breed: {record.breed_name}")
            print(f"o Confidence: {record.confidence_level}%")
            print(f"o Status: {db_status}")

        except Exception as e:
            print(f"Inference Error: {e}")
    else:
        print(f"No valid test images found in {first_breed_dir} to run the inference test.")
