# NYCU Computer Vision 2026 - HW1 📸

## Introduction
This repository contains the source code for Homework 1 of the NYCU Computer Vision course. The goal of this project is to build a robust image classification pipeline for a 100-class dataset. 

To achieve high accuracy, this pipeline utilizes **ResNet101** as the backbone and integrates several advanced deep learning techniques:
* **Advanced Augmentation**: RandAugment, Mixup, and CutMix.
* **Regularization**: Label Smoothing and Dropout.
* **Learning Rate Scheduling**: Cosine Annealing Learning Rate.
* **Inference Optimization**: 10-Crop Test Time Augmentation (TTA).

## Environment Setup
Ensure you have Python 3.8+ installed. The pipeline is built using PyTorch.

**1. Install dependencies**:
```bash
pip install -r requirements.txt
```

**2. Prepare the dataset**:
Please ensure your data directory is structured as follows before running the scripts:
```text
cv_hw1/
├── data/
│   ├── train/       # Training images (organized by class folders)
│   ├── val/         # Validation images (organized by class folders)
│   └── test/        # Test images (flat folder for inference)
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

## Usage

The project is modularized into two main scripts for training and inference.

### 1. Training the Model
To start training the ResNet101 model, simply run:
```bash
python train.py
```
* **What it does**: This script trains the model for 30 epochs (default) using AdamW optimizer and CosineAnnealingLR. It applies Mixup/CutMix dynamically with a 50% probability.
* **Output**: The best model weights will be saved automatically as `best_resnet101.pth`.

### 2. Generating Predictions
Once training is complete, generate the submission file by running:
```bash
python inference.py
```
* **What it does**: This script loads the `best_resnet101.pth` weights and performs **10-Crop TTA (Test Time Augmentation)** on the test images to ensure highly confident and robust predictions.
* **Output**: A `prediction.csv` file will be generated.

## Performance Snapshot

* **Model Architecture**: ResNet101
* **Model Size**: ~42.50 M parameters
* **Key Techniques**: Mixup + CutMix + RandAugment + 10-Crop TTA
