import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_fieldsplit import (
    DATA_ROOT,
    BATCH_SIZE,
    NUM_EPOCHS,
    INPUT_SIZE,
    LR,
    SCHED_PATIENCE,
    SCHED_THRESHOLD,
    SCHED_FACTOR,
    DROPOUT,
    DEFAULT_SEED,
    load_norm_stats,
    make_loaders,
    train_one_epoch,
    eval_loss,
    evaluate_test,
    save_loss_curves,
    save_eval_plots,
)


class AblationClassifier(nn.Module):
    VARIANTS = ("backbone", "fpn_p5", "fpn_multi")

    def __init__(self, variant: str = "fpn_p5", dropout: float = DROPOUT):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}; got {variant!r}")
        self.variant = variant
        self.dropout = dropout
        channels = [16, 32, 64, 128, 256, 512, 1024]
        in_c = 3
        self.blocks = nn.ModuleList()
        for out_c in channels:
            self.blocks.append(self._make_block(in_c, out_c))
            in_c = out_c
        self.lat = nn.ModuleList([nn.Conv2d(c, 256, kernel_size=1) for c in channels])
        if variant in ("backbone", "fpn_p5"):
            self.prediction = nn.Conv2d(256, 1, kernel_size=3, padding=1)
            self.fc = nn.Linear(1, 1)
        else:
            self.multi_levels = (2, 3, 4, 5, 6)
            self.multi_head = nn.Linear(256 * len(self.multi_levels), 1)

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

    def _backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for blk in self.blocks:
            x = F.relu(blk["b1"](blk["c1"](x)))
            x = F.relu(blk["b2"](blk["c2"](x)))
            x = F.relu(blk["b3"](blk["c3"](x)))
            x = blk["pool"](x)
            x = blk["drop"](x)
            feats.append(x)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._backbone(x)
        if self.variant == "backbone":
            p5 = self.lat[4](feats[4])
            out = self.prediction(p5)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = torch.sigmoid(self.fc(out))
            return out
        if self.variant == "fpn_p5":
            p7 = self.lat[6](feats[6])
            p6 = F.interpolate(p7, size=feats[5].shape[2:], mode="nearest") + self.lat[
                5
            ](feats[5])
            p5 = F.interpolate(p6, size=feats[4].shape[2:], mode="nearest") + self.lat[
                4
            ](feats[4])
            out = self.prediction(p5)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = torch.sigmoid(self.fc(out))
            return out
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
        pyr = [p3, p4, p5, p6, p7]
        pooled = [F.adaptive_avg_pool2d(p, (1, 1)).flatten(1) for p in pyr]
        feat = torch.cat(pooled, dim=1)
        out = torch.sigmoid(self.multi_head(feat))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=AblationClassifier.VARIANTS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--max-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--early-stop-patience", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ckpt_dir: Path = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.txt"
    print(
        f"variant={args.variant}  seed={args.seed}  ckpt_dir={ckpt_dir}  max_epochs={args.max_epochs}  early_stop_patience={args.early_stop_patience}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mean, std = load_norm_stats()
    train_loader, val_loader, test_loader = make_loaders(mean, std)
    model = AblationClassifier(variant=args.variant).to(device)
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
            variant=args.variant,
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
        line = f"[{args.variant}] Epoch [{epoch + 1}/{args.max_epochs}] {epoch_min:.2f}min | Train: {tl:.4f} | Val: {vl:.4f}{mark}"
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
    print(f"\n=== [{args.variant}] field-level TEST SET RESULTS ===")
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
