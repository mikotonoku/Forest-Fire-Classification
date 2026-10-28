from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import models, transforms
import timm

# ============================================================================
# PROJECT PATHS, SUPPORTED FORMATS, AND CLASS LABELS
# ============================================================================

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_data"
SPLITS = ROOT / "splits"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = ["Fire", "Smoke", "Non_Fire"]


# ============================================================================
# REPRODUCIBILITY AND DATASET SPLIT CREATION
# ============================================================================

def canonical(name: str) -> str | None:
    value = re.sub(r"[^a-z]", "", name.lower())
    return {"fire": "Fire", "smoke": "Smoke", "nonfire": "Non_Fire", "nofire": "Non_Fire"}.get(value)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_manifest(seed: int, samples_per_class: int = 1000) -> pd.DataFrame:
    if not RAW.exists():
        raise FileNotFoundError(f"Data directory is missing: {RAW}")
    records = []
    for path in RAW.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = next((canonical(parent.name) for parent in path.parents if canonical(parent.name)), None)
        if label:
            records.append({"path": str(path.resolve()), "label": label})
    frame = pd.DataFrame(records).drop_duplicates("path")
    if frame.empty or set(frame.label) != set(CLASSES):
        found = [] if frame.empty else sorted(frame.label.unique())
        raise ValueError(f"Not all required classes were found. Required: {CLASSES}; found: {found}")
    if samples_per_class:
        available = frame.groupby("label").size()
        too_small = available[available < samples_per_class]
        if not too_small.empty:
            raise ValueError(f"Not enough images for a balanced sample of {samples_per_class} per class: {too_small.to_dict()}")
        # One fixed balanced sample is shared by every model and every repeated run.
        frame = frame.groupby("label", group_keys=False).sample(n=samples_per_class, random_state=seed).reset_index(drop=True)
    train, holdout = train_test_split(frame, test_size=.2, random_state=seed, stratify=frame.label)
    valid, test = train_test_split(holdout, test_size=.5, random_state=seed, stratify=holdout.label)
    result = pd.concat([train.assign(split="train"), valid.assign(split="valid"), test.assign(split="test")])
    SPLITS.mkdir(exist_ok=True)
    result.to_csv(SPLITS / "manifest.csv", index=False)
    result.groupby(["split", "label"]).size().unstack(fill_value=0).to_csv(SPLITS / "distribution.csv")
    return result


# ============================================================================
# PYTORCH DATASET AND IMAGE TRANSFORMATIONS
# ============================================================================

class FireDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform): self.frame, self.transform = frame.reset_index(drop=True), transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            return self.transform(image.convert("RGB")), CLASSES.index(row.label), row.path


# ============================================================================
# MODEL BUILDING BLOCKS AND ARCHITECTURE FACTORIES
# ============================================================================

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        # CBAM channel attention: shared MLP on global average and max pooled features.
        channel = torch.sigmoid(self.channel_mlp(torch.mean(x, dim=(2, 3), keepdim=True)) + self.channel_mlp(torch.amax(x, dim=(2, 3), keepdim=True)))
        x = x * channel
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * self.spatial(spatial)


class ResNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
        self.cbam, self.pool, self.head = CBAM(2048), nn.AdaptiveAvgPool2d(1), nn.Linear(2048, len(CLASSES))
    def forward(self, x): return self.head(self.pool(self.cbam(self.features(x))).flatten(1))


def create_model(name: str) -> nn.Module:
    if name == "M1":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        return model
    if name == "M2": return ResNetCBAM()
    choices = {"M3": "convnext_tiny.fb_in22k_ft_in1k", "M4": "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "M5": "coatnet_0_rw_224.sw_in1k"}
    return timm.create_model(choices[name], pretrained=True, num_classes=len(CLASSES))


def freeze_backbone(model: nn.Module, frozen: bool) -> None:
    for p in model.parameters(): p.requires_grad = not frozen
    # Classification heads must always be trainable.
    for attr in ("fc", "classifier", "head"):
        layer = getattr(model, attr, None)
        if layer:
            for p in layer.parameters(): p.requires_grad = True


def transforms_for(augment: bool):
    norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train = [transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(.75, 1.0))]
    if augment: train += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(12), transforms.ColorJitter(.15, .15, .1)]
    train += [transforms.ToTensor(), transforms.Normalize(*norm)]
    test = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(*norm)])
    return transforms.Compose(train), test


# ============================================================================
# TRAINING, VALIDATION, AND TEST EVALUATION
# ============================================================================

@dataclass
class Config:
    epochs: int; batch_size: int; lr: float; patience: int; workers: int; device: str


def evaluate(model, loader, device):
    model.eval(); losses, actual, predicted, paths = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, p in loader:
            x, y = x.to(device), y.to(device); logits = model(x)
            losses.append(criterion(logits, y).item() * len(y)); actual.extend(y.cpu().tolist()); predicted.extend(logits.argmax(1).cpu().tolist()); paths.extend(p)
    return sum(losses) / len(loader.dataset), actual, predicted, paths


# ============================================================================
# VISUALIZATION: LEARNING CURVES, CONFUSION MATRICES, AND GRAD-CAM
# ============================================================================

def draw_history(history, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax, title in [("loss", axes[0], "Loss"), ("acc", axes[1], "Accuracy")]:
        ax.plot(history[f"train_{key}"], label="train"); ax.plot(history[f"valid_{key}"], label="valid"); ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def save_gradcam(model, image_path: str, expected: int, transform, device: str, destination: Path) -> None:
    """Save a Grad-CAM overlay for a convolutional layer of any compared model."""
    target = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)
    if target is None: return
    captured = {}
    def hook(_, __, output):
        captured["activation"] = output
        output.register_hook(lambda grad: captured.setdefault("gradient", grad))
    handle = target.register_forward_hook(hook)
    raw = Image.open(image_path).convert("RGB")
    x = transform(raw).unsqueeze(0).to(device)
    model.eval(); model.zero_grad(set_to_none=True); logits = model(x); predicted = int(logits.argmax(1).item()); logits[0, predicted].backward()
    handle.remove()
    if "gradient" not in captured or captured["activation"].ndim != 4: return
    weights = captured["gradient"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * captured["activation"]).sum(1)).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(raw.size)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(raw); ax.imshow(cam_image, cmap="jet", alpha=.42)
    ax.set_title(f"true: {CLASSES[expected]}, predicted: {CLASSES[predicted]}"); ax.axis("off"); fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def train_one(name, run, frame, output, config, augment=True, unfreeze=True):
    seed = 20260 + run; seed_everything(seed)
    train_tf, eval_tf = transforms_for(augment)
    loaders = {split: DataLoader(FireDataset(frame[frame.split == split], train_tf if split == "train" else eval_tf), batch_size=config.batch_size, shuffle=split == "train", num_workers=config.workers, pin_memory=config.device.startswith("cuda")) for split in ("train", "valid", "test")}
    model = create_model(name).to(config.device); freeze_backbone(model, True)
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.3)
    history = {f"{s}_{m}": [] for s in ("train", "valid") for m in ("loss", "acc")}; best_loss, remaining, best_state = float("inf"), config.patience, None
    for epoch in range(config.epochs):
        if unfreeze and epoch == max(1, config.epochs * 3 // 10):
            freeze_backbone(model, False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr / 10, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.3)
        model.train(); total_loss, correct = 0., 0
        for x, y, _ in tqdm(loaders["train"], desc=f"{name} run {run + 1} epoch {epoch + 1}", leave=False):
            x, y = x.to(config.device), y.to(config.device); optimizer.zero_grad(); logits = model(x); loss = criterion(logits, y); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item()
        valid_loss, y_true, y_pred, _ = evaluate(model, loaders["valid"], config.device)
        history["train_loss"].append(total_loss / len(loaders["train"].dataset)); history["train_acc"].append(correct / len(loaders["train"].dataset)); history["valid_loss"].append(valid_loss); history["valid_acc"].append(accuracy_score(y_true, y_pred)); scheduler.step(valid_loss)
        if valid_loss < best_loss: best_loss, remaining, best_state = valid_loss, config.patience, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: remaining -= 1
        if remaining == 0: break
    model.load_state_dict(best_state); run_dir = output / f"{name}_run{run + 1}"; run_dir.mkdir()
    torch.save(model.state_dict(), run_dir / "weights.pt"); draw_history(history, run_dir / "history.png")
    _, y_true, y_pred, paths = evaluate(model, loaders["test"], config.device)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred); plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(run_dir / "confusion_matrix.png", dpi=160); plt.close()
    pd.DataFrame({"path": paths, "true": [CLASSES[x] for x in y_true], "predicted": [CLASSES[x] for x in y_pred]}).to_csv(run_dir / "test_predictions.csv", index=False)
    # One correct and, when available, one incorrect prediction make the decision process inspectable.
    correct = next((i for i, (truth, pred) in enumerate(zip(y_true, y_pred)) if truth == pred), None)
    wrong = next((i for i, (truth, pred) in enumerate(zip(y_true, y_pred)) if truth != pred), None)
    if correct is not None: save_gradcam(model, paths[correct], y_true[correct], eval_tf, config.device, run_dir / "gradcam_correct.png")
    if wrong is not None: save_gradcam(model, paths[wrong], y_true[wrong], eval_tf, config.device, run_dir / "gradcam_incorrect.png")
    return {"model": name, "run": run + 1, "accuracy": accuracy_score(y_true, y_pred), "precision_macro": precision, "recall_macro": recall, "f1_macro": f1, "best_valid_loss": best_loss, "epochs_completed": len(history["valid_loss"]), "augment": augment, "unfreeze": unfreeze}


# ============================================================================
# RESULTS EXPORT AND AUTOMATIC MARKDOWN REPORT
# ============================================================================

def write_report(results, output, args):
    table = pd.DataFrame(results); primary = table[table["run"] <= args.runs]
    summary = primary.groupby("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].agg(["mean", "std"]).round(4); summary.to_csv(output / "summary.csv")
    best = primary.sort_values(["model", "best_valid_loss", "f1_macro"], ascending=[True, True, False]).groupby("model", as_index=False).first()
    best.to_csv(output / "best_runs.csv", index=False)
    sample_note = "all available images" if args.samples_per_class == 0 else f"a random balanced sample of {args.samples_per_class} images per class"
    text = ["# HNS Project 1 - Results", "", f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "", "## Configuration", "", f"Transfer learning, {args.epochs} epochs, batch size {args.batch_size}, AdamW (lr {args.lr}), and a stratified 80/10/10 split. Data used: {sample_note}. Each model was run {args.runs} times.", "", "## Summary (mean and standard deviation)", "", summary.to_markdown(), "", "## Best run of each model type", "", "`best_runs.csv` identifies the run with the lowest validation loss. Use its `history.png`, `confusion_matrix.png`, and Grad-CAM images for reporting. Test metrics are not used for model selection.", "", best.to_markdown(index=False), "", "## Output files", "", "Each run contains `history.png`, `confusion_matrix.png`, model weights, and test predictions. Interpret results using the variation across all three runs, not only the highest accuracy."]
    text += ["", "## Additional experiments", "", "`runs.csv` also includes one M1 run without supplemental augmentation and one M1 run without progressive backbone unfreezing. Compare them with the corresponding M1 baseline."]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")


# ============================================================================
# COMMAND-LINE INTERFACE AND EXPERIMENT ORCHESTRATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--models", nargs="+", default=["M1", "M2", "M3", "M4", "M5"], choices=["M1", "M2", "M3", "M4", "M5"])
    parser.add_argument("--runs", type=int, default=3); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--patience", type=int, default=4); parser.add_argument("--workers", type=int, default=0); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=1000, help="Balanced random sample per class; use 0 to use every image.")
    args = parser.parse_args(); frame = make_manifest(args.seed, args.samples_per_class)
    print(frame.groupby(["split", "label"]).size().unstack(fill_value=0))
    if args.prepare_only: return
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Device: {device}")
    output = ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S"); output.mkdir(parents=True)
    config = Config(args.epochs, args.batch_size, args.lr, args.patience, args.workers, device); results = []
    for name in args.models:
        for run in range(args.runs): results.append(train_one(name, run, frame, output, config))
    # Required auxiliary comparisons, each on a single reproducible run.
    results.append(train_one("M1", 90, frame, output, config, augment=False, unfreeze=True))
    results.append(train_one("M1", 91, frame, output, config, augment=True, unfreeze=False))
    pd.DataFrame(results).to_csv(output / "runs.csv", index=False); write_report(results, output, args)


if __name__ == "__main__": main()
