import cv2
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
FISH_DIR = Path("C:/Users/Haiqal/Desktop/PRBX/datasets/Fish4Knowledge/fish_image")    # Folder containing species-wise fish images
MASK_DIR = Path("C:/Users/Haiqal/Desktop/PRBX/datasets/Fish4Knowledge/mask_image")    # Folder containing species-wise masks
YOLO_BASE = Path("C:/Users/Haiqal/Desktop/PRBX/datasets/Fish4K-YOLO")                 # Output base directory
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}
CLASS_COUNT = 23                                # Number of species (fish_01 to fish_23)
PREVIEW_COUNT = 20                              # Number of samples to visualize with bounding boxes



# === STEP 1: Generate YOLO Annotations from Mask ===
def generate_annotations_and_images():
    """
    Generate YOLOv8-compatible bounding box annotations from mask images.
    Each fish image is matched to its corresponding mask, and a bounding box is
    computed using contour detection. The results are saved in 'images/all' and 'labels/all'.
    """

    output_images = YOLO_BASE / "images/all"
    output_labels = YOLO_BASE / "labels/all"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    print("Generating YOLO annotations from masks...")
    for i in range(1, CLASS_COUNT + 1):
        fish_folder = FISH_DIR / f"fish_{i:02d}"
        mask_folder = MASK_DIR / f"mask_{i:02d}"

        for fish_file in tqdm(list(fish_folder.glob("*.png")), desc=f"Class {i:02d}"):
            image = cv2.imread(str(fish_file))
            h, w = image.shape[:2]

            # Match corresponding mask file by filename
            mask_file = mask_folder / fish_file.name.replace("fish", "mask")
            if not mask_file.exists():
                continue

            # Load binary mask and find contours
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Use largest contour to define bounding box
            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)

            # Normalize to YOLO format
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h

            # Save image and annotation
            img_out_path = output_images / fish_file.name
            label_out_path = output_labels / (fish_file.stem + ".txt")

            cv2.imwrite(str(img_out_path), image)
            with open(label_out_path, "w") as f:
                f.write(f"{i - 1} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")



# === STEP 2: Split Dataset into Train / Val / Test ===
def split_dataset():
    """
    Randomly split the processed image-label pairs into train, val, and test folders
    according to the SPLIT_RATIOS defined above. This prepares the dataset for YOLOv8 training.
    """

    src_img_dir = YOLO_BASE / "images/all"
    src_lbl_dir = YOLO_BASE / "labels/all"

    image_files = sorted(list(src_img_dir.glob("*.png")))
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(SPLIT_RATIOS['train'] * n)
    n_val = int(SPLIT_RATIOS['val'] * n)

    split_map = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }

    print("\nSplitting into train/val/test sets...")
    for split, files in split_map.items():
        img_out = YOLO_BASE / "images" / split
        lbl_out = YOLO_BASE / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(files, desc=f"{split.title():<5}"):
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out / lbl_path.name)



# === STEP 3: Visual Check of Bounding Boxes ===
def preview_bounding_boxes():
    """
    Generate visual samples of YOLO bounding boxes on fish images to manually inspect
    if the annotations align correctly with the fish. Output is saved in 'preview/' folder.
    """

    print("\nGenerating sample images with bounding boxes for verification...")

    sample_images = list((YOLO_BASE / "images/train").glob("*.png"))[:PREVIEW_COUNT]
    label_dir = YOLO_BASE / "labels/train"
    preview_dir = YOLO_BASE / "preview"
    preview_dir.mkdir(exist_ok=True)

    for img_path in tqdm(sample_images, desc="Preview"):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                class_id, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                # Draw bounding box and class ID
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Class {int(class_id)}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(str(preview_dir / img_path.name), img)



# === MAIN ===
if __name__ == "__main__":
    """
    Main controller function to execute:
    1. Annotation generation from masks
    2. Splitting dataset
    3. Generating sample previews for manual review
    """
    generate_annotations_and_images()
    split_dataset()
    preview_bounding_boxes()
    print("\nAll steps completed! Check 'Fish4K-YOLO/preview/' for bounding box verification.")