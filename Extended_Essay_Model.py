import csv
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import psutil
from ptflops import get_model_complexity_info
import copy
import seaborn as sns
import pandas as pd

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading with train/val split
data_root = "./data"


def get_data_loaders(transform_train, transform_test, batch_size=64, val_split=0.1):
    # Load full training set
    full_train = datasets.CIFAR10(
        root=data_root, train=True, transform=transform_train, download=True
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, transform=transform_test, download=False
    )

    # Split training into train/val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def ensure_reproducibility(seed=42):
    """Complete reproducibility setup"""
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)

    # CUDA algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class FCNN_Small(nn.Module):
    """~1M parameters"""

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(x)


class FCNN_Medium(nn.Module):
    """~2M parameters"""

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(600, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(x)


class FCNN_Large(nn.Module):
    """~4M parameters"""

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(x)


# Keep your CNN architectures (they look good)
class CNN_Small(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.7),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


class CNN_Medium(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.7),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


class CNN_Large(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.7),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


class FCNN_Small_No_Reg(nn.Module):
    """~1M parameters"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(x)


class FCNN_Medium_No_Reg(nn.Module):
    """~2M parameters"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 600),
            nn.ReLU(),
            nn.Linear(600, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(x)


class FCNN_Large_No_Reg(nn.Module):
    """~4M parameters"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 1100),
            nn.ReLU(),
            nn.Linear(1100, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(x)


class CNN_Small_No_Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class CNN_Medium_No_Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class CNN_Large_No_Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops_params(model, input_res=(3, 32, 32)):
    """Compute FLOPs (MACs) and params using ptflops."""
    model_cp = copy.deepcopy(model).to("cpu")
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model_cp,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    return int(macs), int(params)  # MACs per image, params total


def measure_throughput(model, loader, device, warmup_batches=5, measure_batches=20):
    """Measure inference throughput (images/sec)."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            _ = model(x)
            if i + 1 >= warmup_batches:
                break
    # Measure
    start = time.time()
    total_images = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            _ = model(x)
            total_images += x.size(0)
            if i + 1 >= measure_batches:
                break
    elapsed = time.time() - start
    return total_images / elapsed


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(targets)
    avg_loss = total_loss / len(data_loader)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy, all_preds, all_targets


def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", alpha=0.8)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", alpha=0.8)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Accuracy curves
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", alpha=0.8)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # Learning rate
    axes[2].plot(epochs, history["lr"], alpha=0.8)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_curves.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Enhanced confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Raw counts

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    # Normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
    )
    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    optimizer,
    criterion,
    device,
    save_dir,
    scheduler=None,
):
    """Training with validation-based early stopping and final test evaluation."""
    os.makedirs(save_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "train_val_gap": [],
        "epoch_time": [],
        "gpu_memory_mb": [],
        "cpu_memory_mb": [],
        "time_to_acc": {},
    }

    # Track milestones for time-to-accuracy
    acc_milestones = {
        0.70: None,
        0.75: None,
        0.80: None,
        0.85: None,
        0.9: None,
        0.95: None,
    }
    run_start = time.time()

    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation evaluation
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        loss_gap = val_loss - avg_train_loss

        # Save time-to-accuracy milestones
        for thr in list(acc_milestones.keys()):
            if acc_milestones[thr] is None and val_acc >= thr:
                acc_milestones[thr] = time.time() - run_start

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        gpu_mem = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0
        )
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["train_val_gap"].append(loss_gap)
        history["epoch_time"].append(time.time() - epoch_start)
        history["gpu_memory_mb"].append(gpu_mem)
        history["cpu_memory_mb"].append(cpu_mem)
        history["time_to_acc"] = acc_milestones

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Gap: {loss_gap:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"GPU Mem: {gpu_mem:.1f}MB | CPU Mem: {cpu_mem:.1f}MB"
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    # Load best model for final evaluation
    if os.path.exists(os.path.join(save_dir, "best_model.pt")):
        checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])

    # Final unbiased test evaluation
    test_loss, test_acc, test_preds, test_targets = evaluate_model(
        model, test_loader, criterion, device
    )

    # FLOPs + param count
    macs, params_from_flops = compute_flops_params(model)
    param_count = count_parameters(model)
    model_size_mb = param_count * 4 / (1024**2)

    # Inference throughput
    throughput = measure_throughput(model, test_loader, device)

    results = {
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "training_time": training_time,
        "total_epochs": epoch + 1,
        "parameters": param_count,
        "model_size_mb": model_size_mb,
        "macs_inference": macs,
        "flops_inference_est": 2 * macs,
        "inference_images_per_sec": throughput,
        "history": history,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(to_python_types(results), f, indent=2)

    # Save classification report
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    class_report = classification_report(
        test_targets, test_preds, target_names=class_names, output_dict=True
    )
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(class_report, f, indent=2)
    plot_training_curves(history, save_dir)
    plot_confusion_matrix(
        test_targets,
        test_preds,
        class_names,
        os.path.join(save_dir, "confusion_matrix.png"),
    )
    return model, results


# Enhanced augmentation strategies
augmentations = {
    "minimal": {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    },
    "standard": {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    },
}
model_variants = {
    # FCNN models with minimal augmentation
    "fcnn_small_noreg": {"model": FCNN_Small_No_Reg, "augmentation": "minimal"},
    "fcnn_medium_noreg": {"model": FCNN_Medium_No_Reg, "augmentation": "minimal"},
    "fcnn_large_noreg": {"model": FCNN_Large_No_Reg, "augmentation": "minimal"},
    "fcnn_small": {"model": FCNN_Small, "augmentation": "minimal"},
    "fcnn_medium": {"model": FCNN_Medium, "augmentation": "minimal"},
    "fcnn_large": {"model": FCNN_Large, "augmentation": "minimal"},
    # CNN models with standard augmentation
    "cnn_small_noreg": {"model": CNN_Small_No_Reg, "augmentation": "standard"},
    "cnn_medium_noreg": {"model": CNN_Medium_No_Reg, "augmentation": "standard"},
    "cnn_large_noreg": {"model": CNN_Large_No_Reg, "augmentation": "standard"},
    "cnn_small": {"model": CNN_Small, "augmentation": "standard"},
    "cnn_medium": {"model": CNN_Medium, "augmentation": "standard"},
    "cnn_large": {"model": CNN_Large, "augmentation": "standard"},
}


def getParameterCounts():
    for aug_name, aug_transforms in augmentations.items():
        for model_name, model_info in model_variants.items():
            model = model_info["model"]().to(device)
            if model_info["augmentation"] != aug_name:
                continue
            print(model_name + ": " + f"{count_parameters(model):,}")


def save_results_to_csv():
    """Collect all results and save to CSV for Excel plotting"""
    results_data = []

    # Iterate through all result directories
    for model_name in model_variants.keys():
        result_path = f"results/{model_name}/results.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                results = json.load(f)

            # Extract model characteristics
            model_type = "FCNN" if "fcnn" in model_name else "CNN"
            size = (
                "Small"
                if "small" in model_name
                else ("Medium" if "medium" in model_name else "Large")
            )
            has_regularization = "No" if "noreg" in model_name else "Yes"

            # Calculate derived metrics
            params_millions = results["parameters"] / 1_000_000
            training_efficiency = results["final_test_acc"] / (
                results["training_time"] / 60
            )  # acc per minute
            param_efficiency = (
                results["final_test_acc"] / params_millions
            )  # acc per million params

            # Time to accuracy milestones
            time_to_70 = results["history"]["time_to_acc"].get("0.7", None)
            time_to_75 = results["history"]["time_to_acc"].get("0.75", None)
            time_to_80 = results["history"]["time_to_acc"].get("0.8", None)
            time_to_85 = results["history"]["time_to_acc"].get("0.85", None)
            time_to_90 = results["history"]["time_to_acc"].get("0.90", None)

            # Final training metrics
            final_train_loss = (
                results["history"]["train_loss"][-1]
                if results["history"]["train_loss"]
                else None
            )
            final_val_loss = (
                results["history"]["val_loss"][-1]
                if results["history"]["val_loss"]
                else None
            )
            final_train_val_gap = (
                results["history"]["train_val_gap"][-1]
                if results["history"]["train_val_gap"]
                else None
            )

            # Memory usage (average during training)
            avg_gpu_memory = (
                np.mean(results["history"]["gpu_memory_mb"])
                if results["history"]["gpu_memory_mb"]
                else 0
            )
            avg_cpu_memory = (
                np.mean(results["history"]["cpu_memory_mb"])
                if results["history"]["cpu_memory_mb"]
                else 0
            )

            row = {
                # Model identification
                "Model_Name": model_name,
                "Model_Type": model_type,
                "Model_Size": size,
                "Has_Regularization": has_regularization,
                # Performance metrics
                "Test_Accuracy": results["final_test_acc"],
                "Test_Loss": results["final_test_loss"],
                "Best_Val_Accuracy": results["best_val_acc"],
                "Final_Train_Loss": final_train_loss,
                "Final_Val_Loss": final_val_loss,
                "Train_Val_Gap": final_train_val_gap,
                # Model complexity
                "Parameters": results["parameters"],
                "Parameters_Millions": params_millions,
                "Model_Size_MB": results["model_size_mb"],
                "MACs_Inference": results["macs_inference"],
                "FLOPs_Inference": results["flops_inference_est"],
                # Training efficiency
                "Training_Time_Seconds": results["training_time"],
                "Training_Time_Minutes": results["training_time"] / 60,
                "Total_Epochs": results["total_epochs"],
                "Inference_Images_Per_Sec": results["inference_images_per_sec"],
                # Derived efficiency metrics
                "Training_Efficiency_Acc_Per_Min": training_efficiency,
                "Parameter_Efficiency_Acc_Per_MParam": param_efficiency,
                # Time to accuracy milestones
                "Time_to_70_Percent": time_to_70,
                "Time_to_75_Percent": time_to_75,
                "Time_to_80_Percent": time_to_80,
                "Time_to_85_Percent": time_to_85,
                "Time_to_90_Percent": time_to_90,
                # Resource usage
                "Avg_GPU_Memory_MB": avg_gpu_memory,
                "Avg_CPU_Memory_MB": avg_cpu_memory,
            }

            results_data.append(row)

    # Convert to DataFrame and save
    if results_data:
        df = pd.DataFrame(results_data)

        # Sort by model type and size for better organization
        df["Size_Order"] = df["Model_Size"].map({"Small": 1, "Medium": 2, "Large": 3})
        df = df.sort_values(["Model_Type", "Size_Order", "Has_Regularization"])
        df = df.drop("Size_Order", axis=1)

        # Save main results
        csv_path = "results/experiment_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Results saved to: {csv_path}")

        # Create a summary table for key metrics
        summary_df = df[
            [
                "Model_Name",
                "Model_Type",
                "Model_Size",
                "Has_Regularization",
                "Test_Accuracy",
                "Parameters_Millions",
                "Training_Time_Minutes",
                "Training_Efficiency_Acc_Per_Min",
                "Parameter_Efficiency_Acc_Per_MParam",
            ]
        ].copy()

        summary_csv_path = "results/experiment_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"📊 Summary saved to: {summary_csv_path}")

        return df
    else:
        print("❌ No results found to save to CSV")
        return None


def save_training_history_csv():
    """Save detailed training history for all models"""
    all_history_data = []

    for model_name in model_variants.keys():
        result_path = f"results/{model_name}/results.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                results = json.load(f)

            history = results["history"]
            current_model_history = []
            # Create rows for each epoch
            for epoch in range(len(history["train_loss"])):
                row = {
                    "Model_Name": model_name,
                    "Epoch": epoch + 1,
                    "Train_Loss": history["train_loss"][epoch],
                    "Val_Loss": history["val_loss"][epoch],
                    "Val_Accuracy": history["val_acc"][epoch],
                    "Learning_Rate": history["lr"][epoch],
                    "Train_Val_Gap": history["train_val_gap"][epoch],
                    "Epoch_Time_Seconds": history["epoch_time"][epoch],
                    "GPU_Memory_MB": history["gpu_memory_mb"][epoch],
                    "CPU_Memory_MB": history["cpu_memory_mb"][epoch],
                }
                all_history_data.append(row)
                current_model_history.append(row)
            # Save per-model training history CSV
            model_history_df = pd.DataFrame(current_model_history)
            model_history_csv_path = f"results/{model_name}/training_history.csv"
            model_history_df.to_csv(model_history_csv_path, index=False)

    if all_history_data:
        history_df = pd.DataFrame(all_history_data)
        history_csv_path = "results/training_history_all_models.csv"
        history_df.to_csv(history_csv_path, index=False)
        print(f"📈 Training history saved to: {history_csv_path}")
        return history_df
    else:
        print("❌ No training history found to save")
        return None


def run_multiple_seeds_experiment(
    model_class,
    model_name,
    aug_transforms,
    epochs=50,
    num_seeds=7,
    base_save_dir="results",
):
    """Run experiment multiple times with different seeds and average results"""
    seeds = [42, 123, 456, 789, 1011, 1314, 1617]  # 7 different seeds
    all_results = []
    all_histories = []
    print(f"\n🔄 Running {model_name} with {num_seeds} different seeds...")

    for seed_idx, seed in enumerate(seeds):
        if seed_idx >= num_seeds:
            break
        print(f"\n--- Seed {seed_idx+1}/{num_seeds}: {seed} ---")

        # Set reproducibility for this seed
        ensure_reproducibility(seed)

        # Create seed-specific directory
        seed_save_dir = f"{base_save_dir}/{model_name}/seed_{seed}"
        os.makedirs(seed_save_dir, exist_ok=True)

        # Skip if already exists
        if os.path.exists(os.path.join(seed_save_dir, "results.json")):
            print(f"Skipping existing seed experiment: {seed_save_dir}")
            with open(os.path.join(seed_save_dir, "results.json"), "r") as f:
                results = json.load(f)
            all_results.append(results)
            all_histories.append(results["history"])
            continue

        # Get fresh data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            aug_transforms["train"], aug_transforms["test"], batch_size=128
        )

        # Initialize model and training components
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # Train model
        model, results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            save_dir=seed_save_dir,
            scheduler=scheduler,
        )

        all_results.append(results)
        all_histories.append(results["history"])

        print(f"Seed {seed} completed: Test Acc = {results['final_test_acc']:.4f}")

    # Calculate averaged results
    averaged_results = calculate_averaged_results(all_results, all_histories)

    # Save averaged results
    averaged_save_dir = f"{base_save_dir}/{model_name}"
    os.makedirs(averaged_save_dir, exist_ok=True)

    with open(os.path.join(averaged_save_dir, "averaged_results.json"), "w") as f:
        json.dump(to_python_types(averaged_results), f, indent=2)

    # Also save individual seed results summary
    seeds_summary = {
        "seeds": seeds,
        "individual_results": [
            {
                "seed": seeds[i],
                "test_acc": all_results[i]["final_test_acc"],
                "test_loss": all_results[i]["final_test_loss"],
                "best_val_acc": all_results[i]["best_val_acc"],
                "training_time": all_results[i]["training_time"],
                "total_epochs": all_results[i]["total_epochs"],
                "parameters": all_results[i]["parameters"],
            }
            for i in range(len(all_results))
        ],
    }

    with open(os.path.join(averaged_save_dir, "seeds_summary.json"), "w") as f:
        json.dump(to_python_types(seeds_summary), f, indent=2)

    # Plot averaged training curves
    plot_averaged_training_curves(averaged_results["history"], averaged_save_dir)

    return averaged_results


def calculate_averaged_results(all_results, all_histories):
    """Calculate mean and std for all metrics across seeds"""

    # Scalar metrics to average
    scalar_metrics = [
        "final_test_acc",
        "final_test_loss",
        "best_val_acc",
        "training_time",
        "total_epochs",
        "parameters",
        "model_size_mb",
        "macs_inference",
        "flops_inference_est",
        "inference_images_per_sec",
    ]

    averaged = {}

    # Average scalar metrics
    for metric in scalar_metrics:
        values = [r[metric] for r in all_results]
        averaged[f"{metric}_mean"] = np.mean(values)
        averaged[f"{metric}_std"] = np.std(values)
        averaged[f"{metric}_min"] = np.min(values)
        averaged[f"{metric}_max"] = np.max(values)

    # Average history data (epoch-wise)
    history_keys = [
        "train_loss",
        "val_loss",
        "val_acc",
        "lr",
        "train_val_gap",
        "epoch_time",
        "gpu_memory_mb",
        "cpu_memory_mb",
    ]

    averaged_history = {}

    for key in history_keys:
        # Find minimum length across all seeds (in case of early stopping)
        min_length = min(len(h[key]) for h in all_histories)

        # Truncate all histories to minimum length and calculate mean/std
        truncated_histories = [h[key][:min_length] for h in all_histories]

        if truncated_histories:
            averaged_history[f"{key}_mean"] = np.mean(
                truncated_histories, axis=0
            ).tolist()
            averaged_history[f"{key}_std"] = np.std(
                truncated_histories, axis=0
            ).tolist()

    # Time to accuracy milestones
    acc_milestones = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    time_to_acc_averaged = {}

    for milestone in acc_milestones:
        milestone_key = str(milestone)
        times = []
        for h in all_histories:
            time_val = h["time_to_acc"].get(milestone_key, None)
            if time_val is not None:
                times.append(time_val)

        if times:
            time_to_acc_averaged[f"{milestone_key}_mean"] = np.mean(times)
            time_to_acc_averaged[f"{milestone_key}_std"] = np.std(times)
            time_to_acc_averaged[f"{milestone_key}_achieved_count"] = len(times)
        else:
            time_to_acc_averaged[f"{milestone_key}_mean"] = None
            time_to_acc_averaged[f"{milestone_key}_std"] = None
            time_to_acc_averaged[f"{milestone_key}_achieved_count"] = 0

    averaged_history["time_to_acc"] = time_to_acc_averaged
    averaged["history"] = averaged_history

    return averaged


def plot_averaged_training_curves(averaged_history, save_dir):
    """Plot training curves with error bars for averaged results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    train_loss_mean = averaged_history["train_loss_mean"]
    train_loss_std = averaged_history["train_loss_std"]
    val_loss_mean = averaged_history["val_loss_mean"]
    val_loss_std = averaged_history["val_loss_std"]
    val_acc_mean = averaged_history["val_acc_mean"]
    val_acc_std = averaged_history["val_acc_std"]
    lr_mean = averaged_history["lr_mean"]

    epochs = range(1, len(train_loss_mean) + 1)

    # Loss curves with error bars
    axes[0, 0].errorbar(
        epochs,
        train_loss_mean,
        yerr=train_loss_std,
        label="Train Loss",
        alpha=0.8,
        capsize=3,
    )
    axes[0, 0].errorbar(
        epochs, val_loss_mean, yerr=val_loss_std, label="Val Loss", alpha=0.8, capsize=3
    )
    axes[0, 0].set_title("Loss Curves (Mean ± Std)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves with error bars
    axes[0, 1].errorbar(
        epochs,
        val_acc_mean,
        yerr=val_acc_std,
        label="Val Accuracy",
        alpha=0.8,
        capsize=3,
    )
    axes[0, 1].set_title("Validation Accuracy (Mean ± Std)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(epochs, lr_mean, alpha=0.8)
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # Training-validation gap
    gap_mean = averaged_history["train_val_gap_mean"]
    gap_std = averaged_history["train_val_gap_std"]
    axes[1, 1].errorbar(epochs, gap_mean, yerr=gap_std, alpha=0.8, capsize=3)
    axes[1, 1].set_title("Train-Val Loss Gap (Mean ± Std)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss Gap")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "averaged_training_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_averaged_results_to_csv():
    """Save averaged results across all seeds to CSV"""
    results_data = []

    for model_name in model_variants.keys():
        averaged_results_path = f"results/{model_name}/averaged_results.json"

        if os.path.exists(averaged_results_path):
            with open(averaged_results_path, "r") as f:
                results = json.load(f)

            # Extract model characteristics
            model_type = "FCNN" if "fcnn" in model_name else "CNN"
            size = (
                "Small"
                if "small" in model_name
                else ("Medium" if "medium" in model_name else "Large")
            )
            has_regularization = "No" if "noreg" in model_name else "Yes"

            # Extract averaged metrics
            row = {
                "Model_Name": model_name,
                "Model_Type": model_type,
                "Model_Size": size,
                "Has_Regularization": has_regularization,
                # Performance metrics (mean ± std)
                "Test_Accuracy_Mean": results["final_test_acc_mean"],
                "Test_Accuracy_Std": results["final_test_acc_std"],
                "Test_Loss_Mean": results["final_test_loss_mean"],
                "Test_Loss_Std": results["final_test_loss_std"],
                "Best_Val_Accuracy_Mean": results["best_val_acc_mean"],
                "Best_Val_Accuracy_Std": results["best_val_acc_std"],
                # Model complexity
                "Parameters": int(results["parameters_mean"]),
                "Model_Size_MB": results["model_size_mb_mean"],
                # Training metrics
                "Training_Time_Mean": results["training_time_mean"],
                "Training_Time_Std": results["training_time_std"],
                "Total_Epochs_Mean": results["total_epochs_mean"],
                "Total_Epochs_Std": results["total_epochs_std"],
                # Performance range
                "Test_Accuracy_Min": results["final_test_acc_min"],
                "Test_Accuracy_Max": results["final_test_acc_max"],
                # Efficiency metrics
                "Parameter_Efficiency_Mean": results["final_test_acc_mean"]
                / (results["parameters_mean"] / 1_000_000),
                "Training_Efficiency_Mean": results["final_test_acc_mean"]
                / (results["training_time_mean"] / 60),
            }

            results_data.append(row)

    if results_data:
        df = pd.DataFrame(results_data)
        df["Size_Order"] = df["Model_Size"].map({"Small": 1, "Medium": 2, "Large": 3})
        df = df.sort_values(["Model_Type", "Size_Order", "Has_Regularization"])
        df = df.drop("Size_Order", axis=1)

        csv_path = "results/averaged_experiment_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Averaged results saved to: {csv_path}")
        return df
    else:
        print("❌ No averaged results found")
        return None


if __name__ == "__main__":
    ensure_reproducibility()  # Set initial seed
    epochs = 50
    num_seeds = 7
    os.makedirs("results", exist_ok=True)

    for aug_name, aug_transforms in augmentations.items():
        print(f"\n{'='*60}")
        print(f"Testing augmentation: {aug_name.upper()}")
        print(f"{'='*60}")

        for model_name, model_info in model_variants.items():
            if model_info["augmentation"] != aug_name:
                continue

            # Check if averaged results already exist
            averaged_results_path = f"results/{model_name}/averaged_results.json"
            if os.path.exists(averaged_results_path):
                print(f"Skipping existing averaged experiment: {model_name}")
                continue

            print(f"\n🚀 Starting multi-seed experiment: {model_name.upper()}")

            # Run experiment with multiple seeds
            averaged_results = run_multiple_seeds_experiment(
                model_class=model_info["model"],
                model_name=model_name,
                aug_transforms=aug_transforms,
                epochs=epochs,
                num_seeds=num_seeds,
            )

            print(f"✅ {model_name} completed:")
            print(
                f"   Test Acc: {averaged_results['final_test_acc_mean']:.4f} ± {averaged_results['final_test_acc_std']:.4f}"
            )
            print(
                f"   Range: [{averaged_results['final_test_acc_min']:.4f}, {averaged_results['final_test_acc_max']:.4f}]"
            )

    print("\n🎉 All multi-seed experiments completed!")

    # Save averaged results to CSV
    print("\n📊 Saving averaged results to CSV...")
    averaged_df = save_averaged_results_to_csv()

    # Also save the original single-seed results for comparison
    results_df = save_results_to_csv()
    history_df = save_training_history_csv()

    if averaged_df is not None:
        print(f"✅ Experiment complete! Check results/ directory for all outputs.")
        print(f"📈 Key file: results/averaged_experiment_results.csv")
