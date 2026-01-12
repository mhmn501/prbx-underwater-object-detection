import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIGURATION ===
ORIGINAL_FISH_DIR = Path("C:/Users/Haiqal/Desktop/PRBX/datasets/Fish4Knowledge/fish_image")    # Original species-wise fish images
CLASSIFICATION_BASE = Path("Fish4K-Split")      # Output base directory
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}
PREVIEW_COUNT = 5                                        # Number of samples to preview per split


# === STEP 1: Split Dataset into Train / Val / Test ===
def split_dataset():
    """
    Randomly split the original fish species dataset into train, val, and test sets
    according to the SPLIT_RATIOS defined above. This prepares the dataset for EfficientNet training.
    """

    fish_classes = sorted([d.name for d in ORIGINAL_FISH_DIR.iterdir() if d.is_dir()])
    class_distribution = {'train': {}, 'val': {}, 'test': {}}

    print(f"\nFound {len(fish_classes)} classes. Starting dataset split...")

    for fish_class in tqdm(fish_classes, desc="Classes"):
        images = sorted(list((ORIGINAL_FISH_DIR / fish_class).glob("*.*")))
        random.shuffle(images)

        n = len(images)
        n_train = int(SPLIT_RATIOS['train'] * n)
        n_val = int(SPLIT_RATIOS['val'] * n)

        split_map = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, files in split_map.items():
            out_dir = CLASSIFICATION_BASE / split / fish_class
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy(img_path, out_dir / img_path.name)

            # Record class distribution
            class_distribution[split][fish_class] = len(files)

    print("\nDataset splitting completed!")
    return class_distribution


# === STEP 2: Preview Sample Images ===
def preview_samples():
    """
    Generate sample images from train/val/test splits to visually verify
    that dataset splitting and folder organization were successful.
    """

    print("\nGenerating sample previews...")

    for split in ['train', 'val', 'test']:
        split_dir = CLASSIFICATION_BASE / split
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

        preview_dir = CLASSIFICATION_BASE / "preview" / split
        preview_dir.mkdir(parents=True, exist_ok=True)

        for fish_class in classes:
            images = list((split_dir / fish_class).glob("*.*"))
            sample_images = images[:PREVIEW_COUNT]

            for img_path in sample_images:
                img = Image.open(img_path)
                img.save(preview_dir / f"{split}_{fish_class}_{img_path.name}")


# === STEP 3: Save Class Distribution Summary ===
def save_class_distribution(class_distribution):
    """
    Save the class distribution after splitting into a text file for recordkeeping.
    """

    dist_file = CLASSIFICATION_BASE / "class_distribution_summary.txt"
    with open(dist_file, "w") as f:
        total_all = 0
        for split in ['train', 'val', 'test']:
            f.write(f"\n=== {split.upper()} SPLIT ===\n")
            split_total = 0
            for cls, count in sorted(class_distribution[split].items()):
                f.write(f"{cls}: {count} images\n")
                split_total += count
            f.write(f"\nTotal images in {split}: {split_total}\n")
            f.write("-" * 30 + "\n")
            total_all += split_total

        f.write(f"\n=== OVERALL TOTAL IMAGES: {total_all} ===\n")
    print(f"\nClass distribution summary saved to {dist_file}")


# === MAIN ===
if __name__ == "__main__":
    """
    Main controller function to execute:
    1. Splitting dataset into classification folders
    2. Generating sample previews for manual review
    3. Saving class distribution summary to text file
    """
    class_distribution = split_dataset()
    preview_samples()
    save_class_distribution(class_distribution)
    print("\nAll steps completed! Check 'Fish4K-Split/preview/' and 'class_distribution_summary.txt' for results.")
