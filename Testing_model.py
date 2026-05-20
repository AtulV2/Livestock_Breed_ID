import os
import cv2
# Import the new CattleDetector along with existing components
from ml_core import (
    ImageHandler, 
    DataPreprocessor, 
    BreedClassifier, 
    CattleDetector, 
    get_target_breeds, 
    IMG_SIZE, 
    WEIGHTS_PATH
)

# ==========================================
# Main Execution: Two-Step Inference
# ==========================================
if __name__ == "__main__":
    # 1. Dynamically read breed names
    TARGET_BREEDS = get_target_breeds()

    if not TARGET_BREEDS:
        print("CRITICAL ERROR: Dataset directory not found or empty.")
        exit(1)

    # 2. Initialize tools (Including the new Detector)
    try:
        print("Initializing models (this may take a moment)...")
        detector = CattleDetector() # Step 1: Cow Detector
        classifier = BreedClassifier(num_classes=len(TARGET_BREEDS), model_weights_path=WEIGHTS_PATH) # Step 2: Breed Classifier
        preprocessor = DataPreprocessor(target_size=IMG_SIZE)
        image_handler = ImageHandler()
    except Exception as e:
        print(f"Startup Error: {e}")
        exit(1)

    # 3. Prompt for the image path
    print("\n" + "="*50)
    image_path = input("Enter the path to the image: ").strip().strip('\'"')

    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'.")
        exit(1)

    # 4. Run Two-Step Inference
    try:
        # Load the original image
        raw_image = image_handler.load_image(image_path)
        
        print("Step 1: Detecting cattle in image...")
        # Step A: Use YOLO to find and crop the cow
        cropped_cow = detector.detect_and_crop(raw_image)
        
        if cropped_cow is not None:
            print("Step 2: Cattle detected! Classifying breed...")
            
            # Step B: Preprocess only the cropped section
            processed_tensor = preprocessor.preprocess(cropped_cow)
            
            # Step C: Classify breed
            breed, conf = classifier.predict(processed_tensor, class_names=TARGET_BREEDS)
            
            print("\n--- Output ---")
            print(f"File:           {image_path}")
            print(f"Detection:      SUCCESS")
            print(f"Detected Breed: {breed}")
            print(f"Confidence:     {conf}%")
            print("--------------\n")
        else:
            # If YOLO didn't find class 19 (cow)
            print("\n--- Output ---")
            print(f"File:           {image_path}")
            print(f"Detection:      No Cattle Detected")
            print("Action:         Breed classification skipped.")
            print("--------------\n")
            
    except Exception as e:
        print(f"Inference Error: {e}")