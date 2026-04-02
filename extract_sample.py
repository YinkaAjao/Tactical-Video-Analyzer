import os
import random
import shutil
from pathlib import Path
import kagglehub

print("Downloading/Locating dataset via kagglehub... (This might take a moment)")
dataset_path = kagglehub.dataset_download("fahadahmedkhokhar/soccer-view-and-event-score")

# FOLDER SEARCH 
SOURCE_DIR = None
for root, dirs, files in os.walk(dataset_path):
    if any(d.lower() == "corner" for d in dirs):
        SOURCE_DIR = root
        break

if not SOURCE_DIR:
    print("Error: Could not find the tactical event folders inside the downloaded dataset.")
    exit()

print(f" Found the event classes hidden at: {SOURCE_DIR}")

# CONFIGURATION 
TARGET_DIR = "./data"
CLASSES_TO_KEEP = ["Corner", "Free kick", "Gattempt", "Yellow card"]
SAMPLES_PER_CLASS = 500
TRAIN_SPLIT = 0.8  # 80% train, 20% test

def normalize_name(name):
    """Strips spaces, hyphens, underscores, and trailing plural 's' to force a match"""
    clean_name = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if clean_name.endswith('s'):
        clean_name = clean_name[:-1]
    return clean_name

def create_dataset_structure():
    print("Creating lightweight directory structure in ./data/ ...")
    for split in ['train', 'test']:
        for cls in CLASSES_TO_KEEP:
            Path(f"{TARGET_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def sample_and_copy_data():
    create_dataset_structure()
    
    for cls in CLASSES_TO_KEEP:
        # Match folder names using the normalizer
        actual_class_dir = None
        for d in os.listdir(SOURCE_DIR):
            if normalize_name(d) == normalize_name(cls):
                actual_class_dir = os.path.join(SOURCE_DIR, d)
                break
        
        if not actual_class_dir or not os.path.exists(actual_class_dir):
            print(f"Warning: Could not find {cls} folder. Skipping...")
            continue
            
        all_images = [f for f in os.listdir(actual_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(all_images) < SAMPLES_PER_CLASS:
            print(f"Note: {cls} only has {len(all_images)} images. Using all of them.")
            sampled_images = all_images
        else:
            sampled_images = random.sample(all_images, SAMPLES_PER_CLASS)
            
        split_idx = int(len(sampled_images) * TRAIN_SPLIT)
        train_images = sampled_images[:split_idx]
        test_images = sampled_images[split_idx:]
        
        print(f"Processing {cls}: Copying {len(train_images)} train images and {len(test_images)} test images...")
        
        # Copy to train directory
        for img in train_images:
            shutil.copy2(os.path.join(actual_class_dir, img), 
                         os.path.join(TARGET_DIR, 'train', cls, img))
            
        # Copy to test directory
        for img in test_images:
            shutil.copy2(os.path.join(actual_class_dir, img), 
                         os.path.join(TARGET_DIR, 'test', cls, img))

if __name__ == "__main__":
    sample_and_copy_data()
    print("\nData extraction complete!")
    print("The lightweight training data is now ready in the ./data/ directory.")