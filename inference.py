import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

# ==========================================
# Global Configurations
# ==========================================
DATA_DIR = './data'
BATCH_SIZE = 32
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Dataset Definitions
# ==========================================
class TestDataset(Dataset):
    """Dataset class for loading test images for inference."""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# ==========================================
# Model Builder (Required to load weights)
# ==========================================
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet101(weights=None) # Inference doesn't need pretrained weights download
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(DEVICE)

# ==========================================
# Main Inference Logic
# ==========================================
def main():
    print(f"Initializing inference pipeline on device: {DEVICE}")

    # 1. Fetch class names dynamically from train directory
    train_dir = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}. Needed to map class names.")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    # 2. Inference Transforms (10-Crop TTA)
    print("Starting 10-Crop TTA Inference...")
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(
            lambda crops: torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]
            )
        )
    ])

    test_dataset = TestDataset(os.path.join(DATA_DIR, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 3. Load Model
    model = build_model(num_classes=NUM_CLASSES)
    weight_path = 'best_resnet101.pth'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file {weight_path} not found. Please run train.py first.")
        
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    # 4. Generate Predictions
    results = []
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(DEVICE)
            bs, ncrops, c, h, w = images.size()
            
            # Fuse batch and crop dimensions for forward pass
            outputs = model(images.view(-1, c, h, w))
            
            # Average predictions across the 10 crops
            outputs_avg = outputs.view(bs, ncrops, -1).mean(dim=1)
            preds = outputs_avg.argmax(dim=1)

            for name, p in zip(names, preds):
                image_id = os.path.splitext(name)[0]
                pred_label = class_names[p.item()]
                results.append({"image_name": image_id, "pred_label": pred_label})

    # 5. Save to CSV
    output_csv = "prediction.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"{output_csv} successfully generated and ready for upload!")

if __name__ == "__main__":
    main()
