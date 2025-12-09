
# Deep Learning for Underwater Object Detection with a Focus on Fish Species Recognition and Classification

## Project Overview

This project implements three deep learning architectures—YOLOv8, EfficientNet, and Swin Transformer—for classifying fish species from underwater images using the Fish4Knowledge dataset. It includes complete training, evaluation, visualization, and metric logging pipelines for each model, implemented in PyTorch.

---

## Project Structure

```
PRBX/
├── datasets/                        # <--- Not included, see instructions below
│   ├── Fish4Knowledge/              # Original dataset
│   ├── Fish4K-Split/                # For EfficientNet & Swin Transformer
│   └── Fish4K-YOLO/                 # For YOLOv8 training
├── EfficientNet_Implementation/     # Training & evaluation scripts for EfficientNet
├── SwinTransformer_Implementation/  # Training & evaluation scripts for Swin Transformer
├── YOLOv8_Implementation/           # YOLOv8 training, prediction, and visualization
├── environment.yml                  # Conda environment file
```

---

## Dataset Setup

Due to size constraints, the dataset is not included in this repository. Download the dataset from the following link and extract the contents into the `datasets/` folder:

**[Download Dataset from Google Drive](https://drive.google.com/file/d/1edDj1FIqWpUwPX0sf5qH2N-cevOn5yZe/view?usp=drive_link)**

After extraction, your `datasets/` directory should look like:

```
datasets/
├── Fish4Knowledge/
├── Fish4K-Split/
└── Fish4K-YOLO/
```

---

## Environment Setup

1. **Install Anaconda** (if not already):  
   [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

2. **Create and activate virtual environment:**
```bash
conda env create -f environment.yml
conda activate prbx
```

3. **Launch your preferred editor:**
   - Use **Jupyter Notebook** for easy step-by-step exploration, or  
   - Use **VS Code** for full script-based control

---

## Getting Started

You can run and experiment with any of the models below:

### 1. YOLOv8 (Detection)

- Go to `YOLOv8_Implementation/`
- Run or edit `train_yolov8_fish4k.py` to start training

### 2. EfficientNet (Classification)

- Go to `EfficientNet_Implementation/`
- Use `train_efficientnet_fish4k.py` for full training and testing pipeline
- Plots and metrics are auto-saved

### 3. Swin Transformer (Classification)

- Go to `SwinTransformer_Implementation/`
- Use `train_swin_fish4k.py` for training, early stopping, checkpoint saving, and plotting

> Each script is modular and extensively commented. Paths, parameters, and hyperparameters can be changed at the top of each script.

---

## Output Artifacts

- Model checkpoints (`.pth`) will be saved in each implementation folder
- Metrics will be saved as `.csv`
- Evaluation plots and confusion matrices are saved in each implementation folder

---

## Notes

- Ensure that the path to the dataset and the paths in code (e.g., for loading the model, and datasets) are correct based on your local setup.
- Each implementation uses the same Fish4Knowledge dataset but formatted differently.
- Use GPU when available to accelerate training (AMP-enabled for YOLOv8 & Swin).
- Model configurations and environment are reproducible via `environment.yml`.