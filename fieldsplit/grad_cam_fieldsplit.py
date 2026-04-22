import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_fieldsplit import (
    FPNClassifier,
    ImageFolderWithPaths,
    DATA_ROOT,
    INPUT_SIZE,
    BATCH_SIZE,
    load_norm_stats,
    CKPT_DIR,
)


def load_model(checkpoint_path: Path, device: torch.device) -> FPNClassifier:
    model = FPNClassifier().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"]
    if any((k.startswith("module.") for k in state)):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']} (val loss {ckpt.get('best_val_loss', float('nan')):.4f}, seed {ckpt.get('seed', 'unknown')})"
    )
    return model


class GradCAM:

    def __init__(self, model: FPNClassifier):
        self.model = model
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        model.prediction.register_forward_hook(self._fwd)
        model.prediction.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, inp, _out):
        self._act = inp[0].detach()

    def _bwd(self, _m, grad_input, _grad_output):
        self._grad = grad_input[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        prob = self.model(x)
        prob.sum().backward()
        A, G = (self._act, self._grad)
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return (float(prob.item()), cam)


def denormalize(img_tensor: torch.Tensor, mean, std) -> np.ndarray:
    t = img_tensor.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.permute(1, 2, 0).clip(0, 1).numpy()


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = matplotlib.colormaps["jet"]
    heat = cmap(cam)[..., :3]
    return (1 - alpha) * img_rgb + alpha * heat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT_DIR / "best_model_checkpoint.pth"))
    ap.add_argument(
        "--n-per-category",
        type=int,
        default=3,
        help="Number of images per TP/TN/FP/FN category to visualize.",
    )
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
    print(f"Evaluating {args.split}: {len(ds)} images; classes: {ds.classes}")
    model = load_model(Path(args.checkpoint), device)
    print("Pass 1: categorizing predictions...")
    probs = np.zeros(len(ds))
    labels = np.zeros(len(ds), dtype=int)
    paths: list[str] = [None] * len(ds)
    offset = 0
    with torch.no_grad():
        for batch_i, (images, ys, ps) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            out = model(images).cpu().numpy().reshape(-1)
            n = len(out)
            probs[offset : offset + n] = out
            labels[offset : offset + n] = ys.numpy().astype(int)
            for j, p in enumerate(ps):
                paths[offset + j] = p
            offset += n
    pred = (probs >= 0.5).astype(int)
    rng = np.random.default_rng(0)

    def pick(cat_mask: np.ndarray, k: int) -> list[int]:
        idx = np.where(cat_mask)[0]
        if len(idx) == 0:
            return []
        return list(rng.choice(idx, size=min(k, len(idx)), replace=False))

    k = args.n_per_category
    groups = {
        "tp": pick((pred == 1) & (labels == 1), k),
        "tn": pick((pred == 0) & (labels == 0), k),
        "fp": pick((pred == 1) & (labels == 0), k),
        "fn": pick((pred == 0) & (labels == 1), k),
    }
    print(
        f"Selected — TP:{len(groups['tp'])} TN:{len(groups['tn'])} FP:{len(groups['fp'])} FN:{len(groups['fn'])}"
    )
    out_dir = Path(args.checkpoint).parent / "grad_cam"
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_engine = GradCAM(model)
    summary_lines: list[str] = []
    cached: dict[str, list[tuple[np.ndarray, np.ndarray, float, str]]] = {
        g: [] for g in groups
    }
    for cat, indices in groups.items():
        for rank, idx in enumerate(indices):
            image_t, label_t, path_t = ds[idx]
            x = image_t.unsqueeze(0).to(device)
            prob, cam = cam_engine(x)
            img = denormalize(image_t, mean, std)
            blend = overlay(img, cam)
            fname = f"{cat}_{rank}.png"
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img)
            axes[0].set_title(f"Input  ({Path(path_t).name})")
            axes[0].axis("off")
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")
            axes[2].imshow(blend)
            axes[2].set_title(f"Overlay  prob={prob:.3f}  label={int(label_t)}")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=110)
            plt.close(fig)
            cached[cat].append((img, blend, prob, Path(path_t).name))
            summary_lines.append(
                f"{cat:2s} rank={rank}  idx={idx}  prob={prob:.4f}  file={Path(path_t).name}"
            )
    fig, axes = plt.subplots(4, k, figsize=(5 * k, 5 * 4))
    row_labels = {
        "tp": "True Positive",
        "tn": "True Negative",
        "fp": "False Positive",
        "fn": "False Negative",
    }
    for r, cat in enumerate(("tp", "tn", "fp", "fn")):
        for c in range(k):
            ax = axes[r, c] if k > 1 else axes[r]
            if c < len(cached[cat]):
                _, blend, prob, fn_ = cached[cat][c]
                ax.imshow(blend)
                ax.set_title(f"{row_labels[cat]}  p={prob:.3f}\n{fn_}", fontsize=9)
            else:
                ax.set_title(f"{row_labels[cat]}  (n/a)")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "grid.png", dpi=120)
    plt.close(fig)
    with (out_dir / "summary.txt").open("w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(
        f"\nWrote {sum((len(v) for v in cached.values()))} overlays + grid.png to {out_dir}/"
    )


if __name__ == "__main__":
    main()
