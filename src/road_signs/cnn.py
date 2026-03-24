"""PyTorch CNN classifier training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .data import Sample, SignClassificationDataset
from .label_map import LABELS, NUM_CLASSES


class SmallRoadSignCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@dataclass
class CNNResult:
    report: str
    best_val_acc: float
    model_path: Path



def _make_transforms(image_size: int):
    train_t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
        ]
    )
    eval_t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_t, eval_t



def _evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            correct += (pred == yb).sum().item()
            total += yb.numel()

    acc = correct / max(total, 1)
    return y_true, y_pred, acc



def train_and_eval_cnn(
    train_samples: list[Sample],
    val_samples: list[Sample],
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    output_dir: str | Path,
    seed: int = 42,
) -> CNNResult:
    torch.manual_seed(seed)

    train_t, eval_t = _make_transforms(image_size)
    train_ds = SignClassificationDataset(train_samples, image_size=image_size, transforms=train_t)
    val_ds = SignClassificationDataset(val_samples, image_size=image_size, transforms=eval_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallRoadSignCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"CNN epoch {epoch}/{epochs}"):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)

        _, _, val_acc = _evaluate(model, val_loader, device)
        epoch_loss = running_loss / max(len(train_ds), 1)
        print(f"[CNN] epoch={epoch} train_loss={epoch_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, _ = _evaluate(model, val_loader, device)

    labels_present = sorted(set(y_true))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=[LABELS[i + 1] for i in labels_present],
        digits=4,
        zero_division=0,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cnn_best.pt"
    torch.save(model.state_dict(), out_path)

    return CNNResult(report=report, best_val_acc=best_acc, model_path=out_path)
