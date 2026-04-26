import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from backend.config import (
        BATCH_SIZE,
        COMBINED_DATASET_PATH,
        DATA_AUGMENTATION,
        EARLY_STOPPING_PATIENCE,
        EPOCHS,
        LEARNING_RATE,
        LR_SCHEDULER_FACTOR,
        LR_SCHEDULER_PATIENCE,
        MODELS_PATH,
        ensure_dirs,
    )
except ImportError:
    from config import (  # type: ignore
        BATCH_SIZE,
        COMBINED_DATASET_PATH,
        DATA_AUGMENTATION,
        EARLY_STOPPING_PATIENCE,
        EPOCHS,
        LEARNING_RATE,
        LR_SCHEDULER_FACTOR,
        LR_SCHEDULER_PATIENCE,
        MODELS_PATH,
        ensure_dirs,
    )


MODEL_SPECS = {
    "resnet50": {
        "name": "resnet50",
        "input_size": 224,
        "output_file": "resnet50_model.h5",
    },
    "efficientnet": {
        "name": "efficientnet_b3",
        "input_size": 300,
        "output_file": "efficientnet_model.h5",
    },
    "vit": {
        "name": "vit_base_patch16_384",
        "input_size": 384,
        "output_file": "vit_model.h5",
    },
}


def _build_model(model_key: str) -> nn.Module:
    spec = MODEL_SPECS[model_key]
    model = timm.create_model(spec["name"], pretrained=True, num_classes=0, global_pool="avg")
    in_features = model.num_features
    if model_key == "resnet50":
        head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    elif model_key == "efficientnet":
        head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    else:
        head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
    return nn.Sequential(model, head)


def _transforms(input_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_ops: List[transforms.transforms.Compose] = [
        transforms.Resize((input_size, input_size)),
    ]
    if DATA_AUGMENTATION:
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
            ]
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.01),
            transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0)),
        ]
    )
    val_ops = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def _build_loaders(input_size: int) -> Tuple[DataLoader, DataLoader, Dict[int, str], List[int]]:
    train_t, val_t = _transforms(input_size)
    train_ds = datasets.ImageFolder(Path(COMBINED_DATASET_PATH) / "train", transform=train_t)
    val_ds = datasets.ImageFolder(Path(COMBINED_DATASET_PATH) / "validation", transform=val_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    labels = [y for _, y in train_ds.samples]
    return train_loader, val_loader, idx_to_class, labels


def _train_single_model(
    model_key: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    model.to(device)
    best_val = float("inf")
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    output_file = Path(MODELS_PATH) / MODEL_SPECS[model_key]["output_file"]

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += labels.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"[{model_key}] Epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), output_file)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[{model_key}] Early stopping triggered at epoch {epoch + 1}.")
                break

    return history


def train_ensemble() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_histories: Dict[str, Dict[str, List[float]]] = {}
    ensemble_meta = {"models": []}
    class_weights_json: Dict[str, float] = {}

    for model_key in ("resnet50", "efficientnet", "vit"):
        print(f"Training model: {model_key}")
        train_loader, val_loader, idx_to_class, labels = _build_loaders(MODEL_SPECS[model_key]["input_size"])
        classes = np.array(sorted(set(labels)))
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=np.array(labels),
        )
        class_tensor = torch.tensor(class_weights, dtype=torch.float32)
        for cls_idx, weight in zip(classes, class_weights):
            class_weights_json[idx_to_class[cls_idx]] = float(weight)

        model = _build_model(model_key)
        history = _train_single_model(
            model_key=model_key,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_tensor,
            device=device,
        )
        model_histories[model_key] = history
        ensemble_meta["models"].append(
            {
                "name": model_key,
                "architecture": MODEL_SPECS[model_key]["name"],
                "input_size": MODEL_SPECS[model_key]["input_size"],
                "weights_path": str(Path(MODELS_PATH) / MODEL_SPECS[model_key]["output_file"]),
            }
        )

    with (Path(MODELS_PATH) / "ensemble_model.pkl").open("wb") as f:
        pickle.dump(ensemble_meta, f)
    with (Path(MODELS_PATH) / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(model_histories, f, indent=2)
    with (Path(MODELS_PATH) / "class_weights.json").open("w", encoding="utf-8") as f:
        json.dump(class_weights_json, f, indent=2)

    print("Training complete. Saved model weights and ensemble metadata.")


if __name__ == "__main__":
    train_ensemble()