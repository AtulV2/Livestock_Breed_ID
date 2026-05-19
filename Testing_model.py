import os
from ml_core import ImageHandler, DataPreprocessor, BreedClassifier, get_target_breeds, IMG_SIZE, WEIGHTS_PATH

# ==========================================
# Main Execution: Prompt Once & Exit
# ==========================================
if __name__ == "__main__":
    # 1. Dynamically read breed names from your dataset folders
    TARGET_BREEDS = get_target_breeds()

    if not TARGET_BREEDS:
        print("CRITICAL ERROR: Dataset directory not found or empty. Needed to load breed names.")
        exit(1)

    # 2. Initialize tools
    try:
        classifier = BreedClassifier(num_classes=len(TARGET_BREEDS), model_weights_path=WEIGHTS_PATH)
        preprocessor = DataPreprocessor(target_size=IMG_SIZE)
        image_handler = ImageHandler()
    except Exception as e:
        print(f"Startup Error: {e}")
        exit(1)

    # 3. Prompt the user for the path
    print("\n" + "="*50)
    image_path = input("Enter the full or relative path to the image: ").strip()
    
    # Strip accidental quotes if dragged and dropped
    image_path = image_path.strip('\'"')

    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'. Exiting.")
        exit(1)

    # 4. Run inference and exit
    try:
        raw_image = image_handler.load_image(image_path)
        processed_tensor = preprocessor.preprocess(raw_image)
        breed, conf = classifier.predict(processed_tensor, class_names=TARGET_BREEDS)
        
        print("\n--- Output ---")
        print(f"File:           {image_path}")
        print(f"Detected Breed: {breed}")
        print(f"Confidence:     {conf}%")
        print("--------------\n")
        
    except Exception as e:
        print(f"Inference Error: {e}")
