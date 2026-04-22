import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from train_fieldsplit import (
    FPNClassifier,
    NUM_EPOCHS,
    LR,
    INPUT_SIZE,
    SCHED_PATIENCE,
    SCHED_THRESHOLD,
    SCHED_FACTOR,
    DEFAULT_SEED,
    load_norm_stats,
    ImageFolderWithPaths,
    DATA_ROOT,
    eval_loss,
    evaluate_test,
    save_loss_curves,
    save_eval_plots,
)

PHYS_BATCH = 4
ACCUM_STEPS = 4


def make_hardened_loaders(mean: list[float], std: list[float]):
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
        train_ds, batch_size=PHYS_BATCH, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHYS_BATCH, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=PHYS_BATCH, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Classes: {train_ds.classes}")
    print(
        f"Physical batch {PHYS_BATCH} × accum {ACCUM_STEPS} = effective batch {PHYS_BATCH * ACCUM_STEPS}"
    )
    return (train_loader, val_loader, test_loader)


def train_one_epoch_accum(
    model, loader, optimizer, criterion, device, accum_steps: int
) -> float:
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).type_as(outputs))
        (loss / accum_steps).backward()
        total += loss.item()
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total / len(loader)


def enable_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--max-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--early-stop-patience", type=int, default=0)
    args = ap.parse_args()
    enable_determinism(args.seed)
    ckpt_dir: Path = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.txt"
    if torch.cuda.device_count() > 1:
        print(
            f"WARNING: {torch.cuda.device_count()} GPUs visible but hardened recipe requires a single GPU. Re-run with CUDA_VISIBLE_DEVICES=0 to lock to one device."
        )
        print(f"Proceeding on GPU 0 only (DataParallel disabled).")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[hardened] seed={args.seed} ckpt_dir={ckpt_dir} max_epochs={args.max_epochs} early_stop_patience={args.early_stop_patience} device={device}"
    )
    print(
        f"[hardened] cudnn.deterministic={torch.backends.cudnn.deterministic} cudnn.benchmark={torch.backends.cudnn.benchmark}"
    )
    mean, std = load_norm_stats()
    train_loader, val_loader, test_loader = make_hardened_loaders(mean, std)
    model = FPNClassifier().to(device)
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
    print(f"\n[hardened] Training for up to {args.max_epochs} epochs...")
    t0 = time.time()
    for epoch in range(args.max_epochs):
        t = time.time()
        tl = train_one_epoch_accum(
            model, train_loader, optimizer, criterion, device, ACCUM_STEPS
        )
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
            recipe="hardened_single_gpu_cudnn_deterministic",
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
        line = f"[hardened s{args.seed}] Epoch [{epoch + 1}/{args.max_epochs}] {epoch_min:.2f}min | Train: {tl:.4f} | Val: {vl:.4f}{mark}"
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
    print(
        f"\n[hardened] Training complete in {total_min:.1f} min. Best val loss: {best_val:.4f}"
    )
    save_loss_curves(train_losses, val_losses, ckpt_dir / "loss_curves.png")
    best = torch.load(ckpt_dir / "best_model_checkpoint.pth", map_location=device)
    model.load_state_dict(best["state_dict"])
    print(
        f"\nLoaded best model from epoch {best['epoch']} (val loss {best['best_val_loss']:.4f})"
    )
    result = evaluate_test(model, test_loader, device)
    print(f"\n=== [hardened s{args.seed}] field-level TEST SET RESULTS ===")
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
