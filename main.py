import gc
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

# ==========================================
# Global Configurations
# ==========================================
DATA_DIR = './data'
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 100
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
MIXUP_CUTMIX_PROB = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Dataset Definitions
# ==========================================
class TestDataset(Dataset):
    """Dataset class for loading test images for inference.

    Args:
        root_dir (str): Directory containing the test images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

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
# Helper Functions
# ==========================================
def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Generates a random bounding box for CutMix.

    Args:
        size (torch.Size): Size of the image batch (B, C, H, W).
        lam (float): Lambda value determining the ratio of the cut.

    Returns:
        Tuple[int, int, int, int]: Coordinates of the bounding box (x1, y1, x2, y2).
    """
    width, height = size[2], size[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def apply_mixup_cutmix(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies either Mixup or CutMix randomly to a batch of images.

    Args:
        images (torch.Tensor): Batch of input images.
        labels (torch.Tensor): Batch of corresponding labels.
        alpha (float): Alpha parameter for the Beta distribution.

    Returns:
        Tuple containing the mixed images, target A, target B, and lambda.
    """
    rand_val = np.random.rand()
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0]).to(DEVICE)

    target_a = labels
    target_b = labels[rand_index]

    if rand_val < 0.5:
        # Apply Mixup
        mixed_images = lam * images + (1 - lam) * images[rand_index]
    else:
        # Apply CutMix
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        mixed_images = images

    return mixed_images, target_a, target_b, lam


def build_model(num_classes: int) -> nn.Module:
    """Builds the ResNeXt101 model with a custom classification head.

    Args:
        num_classes (int): Number of target classes.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

    # Custom bottleneck classification head with regularization
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> None:
    """Calculates and prints the total number of parameters in the model.

    Args:
        model (nn.Module): The model to evaluate.
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_in_m = total_params / 1_000_000

    print("\n" + "="*40)
    print("Model Parameter Report")
    print("="*40)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size:       {size_in_m:.2f} M")
    
    if size_in_m < 100:
        print("Status: Passed (Under 100M limit).")
    else:
        print("Warning: Exceeded 100M limit!")
    print("="*40 + "\n")


# ==========================================
# Main Execution
# ==========================================
def main():
    print(f"🚀 Initializing training pipeline on device: {DEVICE}")

    # 1. Data Augmentation and Loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. Model Initialization
    model = build_model(num_classes=NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 3. Training Loop
    best_acc = 0.0

    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Apply Mixup/CutMix with probability
            if np.random.rand() < MIXUP_CUTMIX_PROB:
                images, target_a, target_b, lam = apply_mixup_cutmix(images, labels)
                outputs = model(images)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        avg_loss = running_loss / len(train_loader)
        print(f"Validation Acc: {val_acc:.4f} | Training Loss: {avg_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnext101.pth')
            print(f"Saved new best model with accuracy: {best_acc:.4f}")

        scheduler.step()
        gc.collect()

    print(f"Training completed! Best Validation Accuracy: {best_acc:.4f}")

    # 4. Inference and CSV Generation (10-Crop TTA)
    print("\nStarting 10-Crop TTA Inference...")
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
    
    model.load_state_dict(torch.load('best_resnext101.pth'))
    model.eval()

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
                pred_label = train_dataset.classes[p.item()]
                results.append({"image_name": image_id, "pred_label": pred_label})

    # Save to CSV
    pd.DataFrame(results).to_csv("prediction.csv", index=False)
    print("prediction.csv successfully generated and ready for CodaBench upload!")

    # 5. Model Parameter Verification
    count_parameters(model)


if __name__ == "__main__":
    main()
