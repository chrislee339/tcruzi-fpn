import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

MAPPING_CSV = Path(
    "/home/chris/Documents/chagas/audit_package/_mapping_DO_NOT_SHARE.csv"
)
TEST_EVAL = Path(
    "/home/chris/Documents/chagas/checkpoints/hardened_seed69/test_eval.npz"
)


def load_reader(path: Path):
    data = json.loads(path.read_text())
    out = {}
    for k, v in data["scores"].items():
        base = k.rsplit(".", 1)[0]
        out[base] = v.get("answer")
    return (data.get("reader", "unknown"), out)


def load_mapping():
    rows = list(csv.DictReader(MAPPING_CSV.open()))
    by_id = {}
    for r in rows:
        base = r["reader_filename"].rsplit(".", 1)[0]
        by_id[base] = dict(
            audit_category=r["audit_category"],
            original_image=r["original_image"],
            model_prob=float(r["model_prob"]),
            model_pred=int(r["model_pred_at_0_5"]),
            morais_label=int(r["morais_label"]),
            source_path=r["source_path"],
        )
    return by_id


def analyze(path: Path):
    reader, calls = load_reader(path)
    mapping = load_mapping()
    cats = {"CTRL_TP": [], "CTRL_TN": [], "FP": [], "FN": []}
    missing = []
    for base, ans in calls.items():
        if base not in mapping:
            missing.append(base)
            continue
        cats[mapping[base]["audit_category"]].append((base, ans, mapping[base]))
    print(f"\n=== Reader: {reader}  ({path.name}) ===")
    print(f"Total calls scored: {len(calls)}")
    if missing:
        print(
            f"NOTE: {len(missing)} filename(s) in your file are not in the audit mapping: {missing[:3]}"
        )
    print(
        f"\n--- Control calibration (how often you agree with the Morais label on the 40 controls) ---"
    )
    for cat, expected in (("CTRL_TP", "yes"), ("CTRL_TN", "no")):
        rows = cats[cat]
        agree = sum((1 for _, a, _ in rows if a == expected))
        disagree = len(rows) - agree
        label = {
            "CTRL_TP": "Morais = positive, you said:",
            "CTRL_TN": "Morais = negative, you said:",
        }[cat]
        print(
            f"  {cat:8s}  n={len(rows)}   {label}  yes={sum((1 for _, a, _ in rows if a == 'yes'))}  no={sum((1 for _, a, _ in rows if a == 'no'))}   => agree with Morais {agree}/{len(rows)} ({100 * agree / len(rows):.1f}%)"
        )
    print(
        f"\n--- Override pattern on the 44 model-vs-Morais disagreements (how you voted) ---"
    )
    fp_rows = cats["FP"]
    fp_override = sum((1 for _, a, _ in fp_rows if a == "yes"))
    print(
        f"  FP  n={len(fp_rows)}   you sided with the MODEL (said 'yes') on {fp_override}/{len(fp_rows)} ({100 * fp_override / max(1, len(fp_rows)):.1f}%)  — these would flip model FP -> TP if we adopted your label"
    )
    fn_rows = cats["FN"]
    fn_override = sum((1 for _, a, _ in fn_rows if a == "no"))
    print(
        f"  FN  n={len(fn_rows)}   you sided with the MODEL (said 'no') on {fn_override}/{len(fn_rows)} ({100 * fn_override / max(1, len(fn_rows)):.1f}%)  — these would flip model FN -> TN if we adopted your label"
    )
    print(
        f"\n--- Test metrics if we adopted your label as ground truth on ALL 84 audited images ---"
    )
    if not TEST_EVAL.exists():
        print(f"  (skipping — {TEST_EVAL} not found)")
        return
    d = np.load(TEST_EVAL, allow_pickle=True)
    labels = d["labels"].astype(int).copy()
    probs = d["probs"].astype(float)
    if "paths" in d.files:
        paths = [str(p) for p in d["paths"]]
    else:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).parent))
        from train_fieldsplit import ImageFolderWithPaths, DATA_ROOT

        ds = ImageFolderWithPaths(DATA_ROOT / "test")
        paths = [img[0] for img in ds.imgs]
        assert len(paths) == len(
            labels
        ), f"path count {len(paths)} != labels {len(labels)}"
    yhat = (probs >= 0.5).astype(int)
    test_idx_by_orig = {}
    for i, p in enumerate(paths):
        test_idx_by_orig[Path(p).name] = i
    updated = 0
    skipped = 0
    for base, ans, m in cats["FP"] + cats["FN"] + cats["CTRL_TP"] + cats["CTRL_TN"]:
        if ans not in ("yes", "no"):
            skipped += 1
            continue
        new_label = 1 if ans == "yes" else 0
        ti = test_idx_by_orig.get(m["original_image"])
        if ti is None:
            skipped += 1
            continue
        if labels[ti] != new_label:
            labels[ti] = new_label
            updated += 1
    print(
        f"  {updated} of {len(cats['FP']) + len(cats['FN']) + cats['CTRL_TP'].__len__() + cats['CTRL_TN'].__len__()} audited labels changed by your calls; {skipped} had no usable answer."
    )
    tn = int(((yhat == 0) & (labels == 0)).sum())
    fp = int(((yhat == 1) & (labels == 0)).sum())
    tp = int(((yhat == 1) & (labels == 1)).sum())
    fn = int(((yhat == 0) & (labels == 1)).sum())
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    acc = (yhat == labels).mean()
    prec = precision_score(labels, yhat, zero_division=0)
    rec = recall_score(labels, yhat, zero_division=0)
    f1 = f1_score(labels, yhat, zero_division=0)
    spec = tn / (tn + fp) if tn + fp else 0.0
    auc = roc_auc_score(labels, probs)
    print(f"  New confusion at t=0.5: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Accuracy    {acc * 100:6.2f}%   (original 89.52%)")
    print(f"  Precision   {prec * 100:6.2f}%   (original 95.72%)")
    print(f"  Recall      {rec * 100:6.2f}%   (original 88.17%)")
    print(f"  F1          {f1 * 100:6.2f}%   (original 91.79%)")
    print(f"  Specificity {spec * 100:6.2f}%   (original 92.20%)")
    print(f"  AUC         {auc:.4f}          (original 0.9609)")


if __name__ == "__main__":
    paths = (
        [Path(p) for p in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted(
            Path("/home/chris/Documents/chagas/data/scores").glob("scores_*.json")
        )
    )
    if not paths:
        print("No score files provided and none found in data/scores/.")
        sys.exit(1)
    for p in paths:
        analyze(p)
