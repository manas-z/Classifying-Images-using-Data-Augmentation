"""Training script for flower vs. landscape classification with data augmentation."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.datasets import load_sample_images
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DatasetConfig:
    patch_size: int = 64
    train_per_class: int = 80
    val_per_class: int = 200
    seed: int = 42


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    output_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PatchDataset(Dataset):
    """Dataset wrapping pre-generated image patches and labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: transforms.Compose):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])
        pil_image = Image.fromarray(image)
        tensor_image = self.transform(pil_image)
        return tensor_image, label


class SimpleCNN(nn.Module):
    """Three-block convolutional neural network followed by two FC layers."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _sample_patches(image: np.ndarray, num_patches: int, patch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random square patches from an image."""
    h, w, _ = image.shape
    patches = []
    for _ in range(num_patches):
        top = rng.integers(0, h - patch_size + 1)
        left = rng.integers(0, w - patch_size + 1)
        patch = image[top : top + patch_size, left : left + patch_size]
        patches.append(patch)
    return np.stack(patches)


def generate_patch_dataset(config: DatasetConfig) -> Dict[str, np.ndarray]:
    """Generate train/validation patches and labels from sklearn sample images."""
    data = load_sample_images()
    flower_image = data.images[1]  # sunflower image
    landscape_image = data.images[0]  # chinese temple landscape

    rng = np.random.default_rng(config.seed)
    flower_train = _sample_patches(flower_image, config.train_per_class, config.patch_size, rng)
    flower_val = _sample_patches(flower_image, config.val_per_class, config.patch_size, rng)
    landscape_train = _sample_patches(landscape_image, config.train_per_class, config.patch_size, rng)
    landscape_val = _sample_patches(landscape_image, config.val_per_class, config.patch_size, rng)

    train_images = np.concatenate([flower_train, landscape_train], axis=0)
    train_labels = np.concatenate([np.zeros(len(flower_train), dtype=np.int64), np.ones(len(landscape_train), dtype=np.int64)])
    val_images = np.concatenate([flower_val, landscape_val], axis=0)
    val_labels = np.concatenate([np.zeros(len(flower_val), dtype=np.int64), np.ones(len(landscape_val), dtype=np.int64)])

    # Shuffle training and validation sets independently for randomness
    train_perm = rng.permutation(len(train_images))
    val_perm = rng.permutation(len(val_images))
    train_images, train_labels = train_images[train_perm], train_labels[train_perm]
    val_images, val_labels = val_images[val_perm], val_labels[val_perm]

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
    }


def _compute_normalization_stats(images: np.ndarray) -> Tuple[List[float], List[float]]:
    images = images.astype(np.float32) / 255.0
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    std = np.where(std == 0, 1.0, std)
    return mean.tolist(), std.tolist()


def build_datasets(dataset_config: DatasetConfig) -> Dict[str, Dataset]:
    data = generate_patch_dataset(dataset_config)
    mean, std = _compute_normalization_stats(data["train_images"])

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice(
            [
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_baseline = PatchDataset(data["train_images"], data["train_labels"], base_transform)
    train_augmented = PatchDataset(data["train_images"], data["train_labels"], augmentation_transform)
    val_dataset = PatchDataset(data["val_images"], data["val_labels"], base_transform)

    return {
        "baseline": train_baseline,
        "augmented": train_augmented,
        "val": val_dataset,
        "mean": mean,
        "std": std,
    }


def _epoch_metrics_to_dict(epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> Dict[str, float]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc, all_labels, all_preds


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_learning_curves(history: List[Dict[str, float]], output_path: str) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(labels: Iterable[int], preds: Iterable[int], output_path: str, class_names: List[str]) -> None:
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def run_experiment(run_name: str, train_dataset: Dataset, val_dataset: Dataset, training_config: TrainingConfig) -> Dict[str, object]:
    device = torch.device(training_config.device)
    _ensure_output_dir(training_config.output_dir)

    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)

    history: List[Dict[str, float]] = []
    best_val_acc = -math.inf
    best_state: Dict[str, torch.Tensor] | None = None
    best_labels: List[int] = []
    best_preds: List[int] = []

    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, criterion, device)

        history.append(_epoch_metrics_to_dict(epoch, train_loss, train_acc, val_loss, val_acc))
        print(
            f"[{run_name}] Epoch {epoch}/{training_config.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_labels = val_labels
            best_preds = val_preds

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_path = os.path.join(training_config.output_dir, f"{run_name}_metrics.json")
    curves_path = os.path.join(training_config.output_dir, f"{run_name}_learning_curves.png")
    cm_path = os.path.join(training_config.output_dir, f"{run_name}_confusion_matrix.png")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_learning_curves(history, curves_path)
    plot_confusion_matrix(best_labels, best_preds, cm_path, class_names=["flower", "landscape"])

    return {
        "history": history,
        "best_val_accuracy": best_val_acc,
        "metrics_path": metrics_path,
        "curves_path": curves_path,
        "confusion_matrix_path": cm_path,
    }


def run_all_experiments(dataset_config: DatasetConfig | None = None, training_config: TrainingConfig | None = None) -> Dict[str, Dict[str, object]]:
    dataset_config = dataset_config or DatasetConfig()
    training_config = training_config or TrainingConfig()

    set_seed(dataset_config.seed)
    datasets = build_datasets(dataset_config)

    results = {}
    results["baseline"] = run_experiment("baseline", datasets["baseline"], datasets["val"], training_config)
    results["augmented"] = run_experiment("augmented", datasets["augmented"], datasets["val"], training_config)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN baseline and augmented models on flower vs. landscape patches")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to store metrics and plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(seed=args.seed)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    run_all_experiments(dataset_config=dataset_config, training_config=training_config)


if __name__ == "__main__":
    main()
