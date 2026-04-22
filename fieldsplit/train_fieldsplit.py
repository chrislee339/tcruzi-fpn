import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
DATA_ROOT = Path("/home/chris/Documents/chagas/dataset_split_fieldlevel")
CKPT_DIR = Path("/home/chris/Documents/chagas/checkpoints/fieldsplit_v1")
NORM_STATS_FILE = Path("/home/chris/Documents/chagas/fieldsplit/norm_stats.txt")
DEFAULT_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 300
INPUT_SIZE = 1300
LR = 0.0001
SCHED_PATIENCE = 7
SCHED_THRESHOLD = 0.001
SCHED_FACTOR = 0.1
DROPOUT = 0.5
DEFAULT_MEAN = [0.4629, 0.4317, 0.4781]
DEFAULT_STD = [0.4031, 0.3864, 0.4044]


def load_norm_stats() -> tuple[list[float], list[float]]:
    if not NORM_STATS_FILE.exists():
        print(f"[norm] {NORM_STATS_FILE} not found; using paper defaults.")
        return (DEFAULT_MEAN, DEFAULT_STD)
    mean = std = None
    with NORM_STATS_FILE.open() as f:
        for line in f:
            if line.startswith("mean"):
                mean = eval(line.split("=", 1)[1].strip())
            elif line.startswith("std"):
                std = eval(line.split("=", 1)[1].strip())
    if mean is None or std is None:
        print(f"[norm] could not parse {NORM_STATS_FILE}; using paper defaults.")
        return (DEFAULT_MEAN, DEFAULT_STD)
    print(f"[norm] loaded from {NORM_STATS_FILE}: mean={mean}, std={std}")
    return (mean, std)


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return (image, label, path)


class FPNClassifier(nn.Module):

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = dropout
        channels = [16, 32, 64, 128, 256, 512, 1024]
        in_c = 3
        self.blocks = nn.ModuleList()
        for out_c in channels:
            self.blocks.append(self._make_block(in_c, out_c))
            in_c = out_c
        self.lat = nn.ModuleList([nn.Conv2d(c, 256, kernel_size=1) for c in channels])
        self.prediction = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(1, 1)

    def _make_block(self, in_c: int, out_c: int) -> nn.ModuleDict:
        return nn.ModuleDict(
            dict(
                c1=nn.Conv2d(in_c, out_c, 3, 1, 1),
                b1=nn.BatchNorm2d(out_c),
                c2=nn.Conv2d(out_c, out_c, 3, 1, 1),
                b2=nn.BatchNorm2d(out_c),
                c3=nn.Conv2d(out_c, out_c, 3, 1, 1),
                b3=nn.BatchNorm2d(out_c),
                pool=nn.MaxPool2d(2, 2),
                drop=nn.Dropout(self.dropout),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: list[torch.Tensor] = []
        for blk in self.blocks:
            x = F.relu(blk["b1"](blk["c1"](x)))
            x = F.relu(blk["b2"](blk["c2"](x)))
            x = F.relu(blk["b3"](blk["c3"](x)))
            x = blk["pool"](x)
            x = blk["drop"](x)
            feats.append(x)
        p7 = self.lat[6](feats[6])
        p6 = F.interpolate(p7, size=feats[5].shape[2:], mode="nearest") + self.lat[5](
            feats[5]
        )
        p5 = F.interpolate(p6, size=feats[4].shape[2:], mode="nearest") + self.lat[4](
            feats[4]
        )
        p4 = F.interpolate(p5, size=feats[3].shape[2:], mode="nearest") + self.lat[3](
            feats[3]
        )
        p3 = F.interpolate(p4, size=feats[2].shape[2:], mode="nearest") + self.lat[2](
            feats[2]
        )
        p2 = F.interpolate(p3, size=feats[1].shape[2:], mode="nearest") + self.lat[1](
            feats[1]
        )
        p1 = F.interpolate(p2, size=feats[0].shape[2:], mode="nearest") + self.lat[0](
            feats[0]
        )
        out = self.prediction(p5)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = torch.sigmoid(self.fc(out))
        return out


def make_loaders(mean: list[float], std: list[float]):
    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_ds = ImageFolderWithPaths(DATA_ROOT / "train", transform=transform)
    val_ds = ImageFolderWithPaths(DATA_ROOT / "val", transform=transform)
    test_ds = ImageFolderWithPaths(DATA_ROOT / "test", transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Classes: {train_ds.classes}")
    return (train_loader, val_loader, test_loader)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).type_as(outputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).type_as(outputs))
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_test(model, loader, device) -> dict:
    model.eval()
    ys, yhats, probs = ([], [], [])
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        p = outputs.cpu().numpy().reshape(-1)
        probs.extend(p.tolist())
        yhats.extend((p >= 0.5).astype(int).tolist())
        ys.extend(labels.cpu().numpy().astype(int).tolist())
    ys = np.array(ys)
    yhats = np.array(yhats)
    probs = np.array(probs)
    cm = confusion_matrix(ys, yhats)
    tn, fp, fn, tp = cm.ravel()
    return dict(
        cm=cm,
        accuracy=float(accuracy_score(ys, yhats)),
        precision=float(precision_score(ys, yhats, zero_division=0)),
        recall=float(recall_score(ys, yhats, zero_division=0)),
        f1=float(f1_score(ys, yhats, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        auc=float(roc_auc_score(ys, probs)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        probs=probs,
        labels=ys,
    )


def save_loss_curves(train_losses, val_losses, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    skip = min(10, len(train_losses) // 5) if train_losses else 0
    axes[1].plot(range(skip, len(train_losses)), train_losses[skip:], label="Train")
    axes[1].plot(range(skip, len(val_losses)), val_losses[skip:], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"Loss Curves (epoch {skip}+, zoomed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_eval_plots(result: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        result["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        ax=axes[0],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (field-level test)")
    fpr, tpr, _ = roc_curve(result["labels"], result["probs"])
    axes[1].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {result['auc']:.3f})"
    )
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC (field-level test)")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--max-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val loss has not improved for this many epochs (0 disables).",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ckpt_dir: Path = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.txt"
    print(
        f"seed={args.seed}  ckpt_dir={ckpt_dir}  max_epochs={args.max_epochs}  early_stop_patience={args.early_stop_patience}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mean, std = load_norm_stats()
    train_loader, val_loader, test_loader = make_loaders(mean, std)
    model = FPNClassifier().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    print(f"Parameters: {sum((p.numel() for p in model.parameters())):,}")
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE,
        threshold=SCHED_THRESHOLD,
        threshold_mode="rel",
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    epochs_since_improvement = 0
    print(f"\nTraining for up to {args.max_epochs} epochs...")
    t0 = time.time()
    for epoch in range(args.max_epochs):
        t = time.time()
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = eval_loss(model, val_loader, criterion, device)
        train_losses.append(tl)
        val_losses.append(vl)
        epoch_min = (time.time() - t) / 60
        ckpt = dict(
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val,
            seed=args.seed,
        )
        mark = ""
        if vl < best_val - SCHED_THRESHOLD:
            best_val = vl
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, ckpt_dir / "best_model_checkpoint.pth")
            mark = " *BEST*"
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        torch.save(ckpt, ckpt_dir / "recent_model_checkpoint.pth")
        line = f"Epoch [{epoch + 1}/{args.max_epochs}] {epoch_min:.2f}min | Train: {tl:.4f} | Val: {vl:.4f}{mark}"
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")
        scheduler.step(vl)
        if (
            args.early_stop_patience > 0
            and epochs_since_improvement >= args.early_stop_patience
        ):
            print(
                f"\nEarly stop: no improvement for {args.early_stop_patience} epochs."
            )
            break
    total_min = (time.time() - t0) / 60
    print(f"\nTraining complete in {total_min:.1f} min. Best val loss: {best_val:.4f}")
    save_loss_curves(train_losses, val_losses, ckpt_dir / "loss_curves.png")
    best = torch.load(ckpt_dir / "best_model_checkpoint.pth", map_location=device)
    model.load_state_dict(best["state_dict"])
    print(
        f"\nLoaded best model from epoch {best['epoch']} (val loss {best['best_val_loss']:.4f})"
    )
    result = evaluate_test(model, test_loader, device)
    print(f"\n=== Field-level TEST SET RESULTS ===")
    print(f"Accuracy:    {result['accuracy']:.4f}")
    print(f"Precision:   {result['precision']:.4f}")
    print(f"Recall:      {result['recall']:.4f}")
    print(f"F1:          {result['f1']:.4f}")
    print(f"Specificity: {result['specificity']:.4f}")
    print(f"AUC:         {result['auc']:.4f}")
    print(f"TN={result['tn']}  FP={result['fp']}")
    print(f"FN={result['fn']}  TP={result['tp']}")
    print(f"Confusion matrix:\n{result['cm']}")
    np.savez(
        ckpt_dir / "test_eval.npz",
        labels=result["labels"],
        probs=result["probs"],
        cm=result["cm"],
    )
    save_eval_plots(result, ckpt_dir / "eval_plots.png")
    print(f"\nArtifacts in {ckpt_dir}/")


if __name__ == "__main__":
    main()
