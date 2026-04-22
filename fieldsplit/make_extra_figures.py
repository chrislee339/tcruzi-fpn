from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

CKPT_DIR = Path("/home/chris/Documents/chagas/checkpoints/hardened_seed69")
OUT_DIR = Path("/home/chris/Documents/chagas/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
data = np.load(CKPT_DIR / "test_eval.npz", allow_pickle=True)
labels = data["labels"].astype(int)
probs = data["probs"].astype(float)
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(0, 1, 41)
ax.hist(
    probs[labels == 0],
    bins=bins,
    alpha=0.6,
    label=f"negative (n={int((labels == 0).sum())})",
    color="#4c72b0",
)
ax.hist(
    probs[labels == 1],
    bins=bins,
    alpha=0.6,
    label=f"positive (n={int((labels == 1).sum())})",
    color="#c44e52",
)
ax.axvline(0.5, color="k", linestyle="--", lw=1, label="threshold = 0.5")
ax.set_xlabel("Predicted probability of positive class")
ax.set_ylabel("Image count")
ax.set_title("Test-set probability distribution by true class")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_prob_histogram_fieldsplit.png", dpi=150)
plt.close(fig)
thresholds = np.linspace(0.01, 0.99, 99)
prec, rec, f1s, spec = ([], [], [], [])
for t in thresholds:
    yhat = (probs >= t).astype(int)
    prec.append(precision_score(labels, yhat, zero_division=0))
    rec.append(recall_score(labels, yhat, zero_division=0))
    f1s.append(f1_score(labels, yhat, zero_division=0))
    tn = int(((yhat == 0) & (labels == 0)).sum())
    fp = int(((yhat == 1) & (labels == 0)).sum())
    spec.append(tn / (tn + fp) if tn + fp else 0.0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, prec, label="Precision", color="#4c72b0", lw=2)
ax.plot(thresholds, rec, label="Recall (sensitivity)", color="#c44e52", lw=2)
ax.plot(thresholds, spec, label="Specificity", color="#55a868", lw=2)
ax.plot(thresholds, f1s, label="F1", color="#8172b2", lw=2, linestyle="--")
ax.axvline(0.5, color="k", lw=1, linestyle=":")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Metric value")
ax.set_title("Metric vs decision threshold on the test set")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower center", ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_threshold_sweep_fieldsplit.png", dpi=150)
plt.close(fig)
p, r, _ = precision_recall_curve(labels, probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(r, p, color="#c44e52", lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-recall curve (field-level test set)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_pr_curve_fieldsplit.png", dpi=150)
plt.close(fig)
print("Wrote:")
for p in sorted(OUT_DIR.glob("fig_*_fieldsplit.png")):
    print(f"  {p}")
