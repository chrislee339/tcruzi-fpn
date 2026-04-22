import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
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
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_fieldsplit import (
    FPNClassifier,
    ImageFolderWithPaths,
    DATA_ROOT,
    NORM_STATS_FILE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    INPUT_SIZE,
    BATCH_SIZE,
    load_norm_stats,
    CKPT_DIR,
)


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = FPNClassifier().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"]
    has_module_prefix = any((k.startswith("module.") for k in state.keys()))
    needs_module_prefix = isinstance(model, nn.DataParallel)
    if has_module_prefix and (not needs_module_prefix):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    elif not has_module_prefix and needs_module_prefix:
        state = {f"module.{k}": v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']} (val loss {ckpt.get('best_val_loss', float('nan')):.4f})"
    )
    return model


@torch.no_grad()
def run(model: nn.Module, loader: DataLoader, device: torch.device):
    ys, probs, paths = ([], [], [])
    for images, labels, ps in loader:
        images = images.to(device, non_blocking=True)
        out = model(images).cpu().numpy().reshape(-1)
        probs.extend(out.tolist())
        ys.extend(labels.numpy().astype(int).tolist())
        paths.extend(list(ps))
    return (np.array(ys), np.array(probs), paths)


def metrics_at(ys, probs, threshold: float) -> dict:
    yhat = (probs >= threshold).astype(int)
    cm = confusion_matrix(ys, yhat)
    tn, fp, fn, tp = cm.ravel()
    return dict(
        threshold=threshold,
        accuracy=float(accuracy_score(ys, yhat)),
        precision=float(precision_score(ys, yhat, zero_division=0)),
        recall=float(recall_score(ys, yhat, zero_division=0)),
        f1=float(f1_score(ys, yhat, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        cm=cm.tolist(),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT_DIR / "best_model_checkpoint.pth"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm_stats()
    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    ds = ImageFolderWithPaths(DATA_ROOT / args.split, transform=transform)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Evaluating on {args.split}: {len(ds)} images; classes: {ds.classes}")
    model = load_model(Path(args.checkpoint), device)
    ys, probs, paths = run(model, loader, device)
    auc = float(roc_auc_score(ys, probs))
    m = metrics_at(ys, probs, args.threshold)
    print(f"\n=== {args.split} metrics at threshold {args.threshold:.2f} ===")
    print(f"Accuracy:    {m['accuracy']:.4f}")
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall:      {m['recall']:.4f}")
    print(f"F1:          {m['f1']:.4f}")
    print(f"Specificity: {m['specificity']:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"Confusion matrix:\n{np.array(m['cm'])}")
    out_dir = Path(args.checkpoint).parent
    np.savez(
        out_dir / f"{args.split}_eval.npz",
        labels=ys,
        probs=probs,
        paths=np.array(paths),
    )
    with (out_dir / f"{args.split}_metrics.json").open("w") as f:
        json.dump(dict(auc=auc, **m), f, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = np.array(m["cm"])
    sns.heatmap(
        cm,
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
    axes[0].set_title(f"{args.split} confusion matrix @ {args.threshold:.2f}")
    fpr, tpr, _ = roc_curve(ys, probs)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title(f"{args.split} ROC")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.split}_eval.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {args.split}_eval.npz / _metrics.json / _eval.png in {out_dir}/")


if __name__ == "__main__":
    main()
