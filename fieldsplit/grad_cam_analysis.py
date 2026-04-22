import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
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
from grad_cam_fieldsplit import GradCAM, load_model

MORAIS_CSV = Path("/home/chris/Documents/chagas/peerj-10-13470-s001_all.csv")
RESIZED_W, RESIZED_H = (1224, 1632)
QW, QH = (RESIZED_W // 2, RESIZED_H // 2)
QUAD_ORIGIN = {1: (0, 0), 2: (QW, 0), 3: (0, QH), 4: (QW, QH)}


def load_morais_annotations() -> dict[str, list[tuple[float, float]]]:
    per_field: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    with MORAIS_CSV.open() as f:
        r = csv.DictReader(f)
        for row in r:
            field = row["filename"].replace(".jpg", "")
            per_field[field].append(
                (row["structure"], float(row["x"]), float(row["y"]))
            )
    out: dict[str, list[tuple[float, float]]] = {}
    for field, items in per_field.items():
        centroids: list[tuple[float, float]] = []
        i = 0
        while i < len(items):
            s1, x1, y1 = items[i]
            if i + 1 < len(items):
                s2, x2, y2 = items[i + 1]
                if {s1, s2} == {"NUCLEUS", "KINETOPLAST"}:
                    centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
                    i += 2
                    continue
            centroids.append((x1, y1))
            i += 1
        out[field] = centroids
    return out


def _load_size_table() -> dict[str, tuple[int, int]]:
    import csv, os
    from collections import defaultdict

    ann_rows = list(csv.DictReader(MORAIS_CSV.open()))
    per_field: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in ann_rows:
        fn = r["filename"].replace(".jpg", "")
        per_field[fn].append((float(r["x"]), float(r["y"])))
    from pathlib import Path as _P

    split_root = _P("/home/chris/Documents/chagas/dataset_split_fieldlevel")
    field_pos_quads: dict[str, set[int]] = defaultdict(set)
    for split in ("train", "val", "test"):
        for cls, lbl in (("positive_images", 1), ("negative_images", 0)):
            d = split_root / split / cls
            if not d.is_dir():
                continue
            for fn in os.listdir(d):
                if not fn.startswith("field") or fn.startswith("noisy_"):
                    continue
                base = fn.replace(".png", "").split("_")
                if len(base) < 2 or not base[0].startswith("field"):
                    continue
                try:
                    q = int(base[1].replace("quad", ""))
                except ValueError:
                    continue
                if lbl == 1:
                    field_pos_quads[base[0]].add(q)

    def assign_q(x, y, W, H):
        QW, QH = (W // 2, H // 2)
        if x < QW and y < QH:
            return 1
        elif x >= QW and y < QH:
            return 2
        elif x < QW and y >= QH:
            return 3
        else:
            return 4

    sizes = {}
    for field, pts in per_field.items():
        expected = field_pos_quads.get(field)
        if not expected:
            sizes[field] = (2448, 3264)
            continue
        qs_small = {assign_q(x, y, 2448, 3264) for x, y in pts}
        qs_big = {assign_q(x, y, 3456, 4608) for x, y in pts}
        if qs_small == expected:
            sizes[field] = (2448, 3264)
        elif qs_big == expected:
            sizes[field] = (3456, 4608)
        else:
            sizes[field] = (
                (2448, 3264)
                if len(qs_small & expected) >= len(qs_big & expected)
                else (3456, 4608)
            )
    return sizes


_FIELD_SIZE: dict[str, tuple[int, int]] | None = None


def original_field_size(field_base: str) -> tuple[int, int]:
    global _FIELD_SIZE
    if _FIELD_SIZE is None:
        _FIELD_SIZE = _load_size_table()
    return _FIELD_SIZE.get(field_base, (2448, 3264))


def filename_to_quadrant(fn: str) -> tuple[str, int]:
    m = re.fullmatch("(field\\d+)_quad([1-4])\\.png", fn)
    if not m:
        raise ValueError(f"can't parse quadrant from {fn!r}")
    return (m.group(1), int(m.group(2)))


def project_annotation_to_quadrant(
    cx: float, cy: float, orig_w: int, orig_h: int, quad: int
) -> tuple[bool, float, float]:
    x_r = cx * RESIZED_W / orig_w
    y_r = cy * RESIZED_H / orig_h
    x0, y0 = QUAD_ORIGIN[quad]
    if not (x0 <= x_r < x0 + QW and y0 <= y_r < y0 + QH):
        return (False, -1.0, -1.0)
    x_q = x_r - x0
    y_q = y_r - y0
    x_m = x_q * INPUT_SIZE / QW
    y_m = y_q * INPUT_SIZE / QH
    return (True, x_m, y_m)


def compute_cam_stats(cam: np.ndarray) -> dict:
    h, w = cam.shape
    total = cam.sum()
    if total <= 0:
        return dict(
            peak_x=0.5,
            peak_y=0.5,
            com_x=0.5,
            com_y=0.5,
            boundary_fraction=0.0,
            max_value=0.0,
        )
    r, c = np.unravel_index(np.argmax(cam), cam.shape)
    yy = np.arange(h).reshape(-1, 1)
    xx = np.arange(w).reshape(1, -1)
    com_x = float((cam * xx).sum() / total) / (w - 1)
    com_y = float((cam * yy).sum() / total) / (h - 1)
    margin_r = max(1, int(round(h * 0.1)))
    margin_c = max(1, int(round(w * 0.1)))
    edge_mass = (
        cam[:margin_r, :].sum()
        + cam[-margin_r:, :].sum()
        + cam[:, :margin_c].sum()
        + cam[:, -margin_c:].sum()
        - cam[:margin_r, :margin_c].sum()
        - cam[:margin_r, -margin_c:].sum()
        - cam[-margin_r:, :margin_c].sum()
        - cam[-margin_r:, -margin_c:].sum()
    )
    return dict(
        peak_x=float(c) / (w - 1),
        peak_y=float(r) / (h - 1),
        com_x=com_x,
        com_y=com_y,
        boundary_fraction=float(edge_mass / total),
        max_value=float(cam.max()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT_DIR / "best_model_checkpoint.pth"))
    ap.add_argument(
        "--hit-radius-px",
        type=float,
        default=150.0,
        help="Distance threshold (in 1300-pixel units) for 'peak near parasite'.",
    )
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
    ds = ImageFolderWithPaths(DATA_ROOT / "test", transform=transform)
    print(f"Test images: {len(ds)}; classes: {ds.classes}")
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    model = load_model(Path(args.checkpoint), device)
    print("Pass 1: collecting predictions...")
    probs = np.zeros(len(ds))
    labels = np.zeros(len(ds), dtype=int)
    paths: list[str] = [None] * len(ds)
    offset = 0
    with torch.no_grad():
        for images, ys, ps in loader:
            images = images.to(device, non_blocking=True)
            out = model(images).cpu().numpy().reshape(-1)
            n = len(out)
            probs[offset : offset + n] = out
            labels[offset : offset + n] = ys.numpy().astype(int)
            for j, p in enumerate(ps):
                paths[offset + j] = p
            offset += n
    pred = (probs >= 0.5).astype(int)
    print("Pass 2: Grad-CAM per image...")
    cam_engine = GradCAM(model)
    stats: list[dict] = []
    for i in range(len(ds)):
        image_t, _, path_t = ds[i]
        x = image_t.unsqueeze(0).to(device)
        _, cam = cam_engine(x)
        s = compute_cam_stats(cam)
        s["index"] = i
        s["path"] = path_t
        s["prob"] = float(probs[i])
        s["label"] = int(labels[i])
        s["pred"] = int(pred[i])
        stats.append(s)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}")

    def cat(s):
        if s["pred"] == 1 and s["label"] == 1:
            return "tp"
        if s["pred"] == 0 and s["label"] == 0:
            return "tn"
        if s["pred"] == 1 and s["label"] == 0:
            return "fp"
        return "fn"

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in stats:
        groups[cat(s)].append(s)
    summary = {}
    for c in ("tp", "tn", "fp", "fn"):
        arr = groups[c]
        if not arr:
            continue
        summary[c] = dict(
            n=len(arr),
            mean_prob=float(np.mean([x["prob"] for x in arr])),
            mean_boundary_fraction=float(
                np.mean([x["boundary_fraction"] for x in arr])
            ),
            mean_peak_x=float(np.mean([x["peak_x"] for x in arr])),
            mean_peak_y=float(np.mean([x["peak_y"] for x in arr])),
            mean_com_x=float(np.mean([x["com_x"] for x in arr])),
            mean_com_y=float(np.mean([x["com_y"] for x in arr])),
        )
    ann = load_morais_annotations()
    near_parasite_hits = 0
    tp_with_annotations = 0
    tp_distances: list[float] = []
    near_parasite_lines: list[str] = []
    for s in groups["tp"]:
        fn = Path(s["path"]).name
        field, quad = filename_to_quadrant(fn)
        cents = ann.get(field, [])
        orig_w, orig_h = original_field_size(field)
        in_quad: list[tuple[float, float]] = []
        for cx, cy in cents:
            ok, x_m, y_m = project_annotation_to_quadrant(cx, cy, orig_w, orig_h, quad)
            if ok:
                in_quad.append((x_m, y_m))
        if not in_quad:
            continue
        tp_with_annotations += 1
        peak_x_px = s["peak_x"] * (INPUT_SIZE - 1)
        peak_y_px = s["peak_y"] * (INPUT_SIZE - 1)
        dists = [
            ((peak_x_px - px) ** 2 + (peak_y_px - py) ** 2) ** 0.5 for px, py in in_quad
        ]
        dmin = min(dists)
        tp_distances.append(dmin)
        hit = dmin <= args.hit_radius_px
        near_parasite_hits += int(hit)
        near_parasite_lines.append(
            f"{fn:30s}  prob={s['prob']:.3f}  peak=({peak_x_px:.0f},{peak_y_px:.0f})  nearest_parasite=({in_quad[int(np.argmin(dists))][0]:.0f},{in_quad[int(np.argmin(dists))][1]:.0f})  dist={dmin:.0f}px  hit={hit}"
        )
    summary["tp_localization"] = dict(
        tp_total=len(groups["tp"]),
        tp_with_annotations=tp_with_annotations,
        hit_radius_px=args.hit_radius_px,
        hits=near_parasite_hits,
        hit_rate=float(near_parasite_hits) / max(1, tp_with_annotations),
        mean_dist_px=float(np.mean(tp_distances)) if tp_distances else None,
        median_dist_px=float(np.median(tp_distances)) if tp_distances else None,
    )
    out_dir = Path(args.checkpoint).parent
    with (out_dir / "grad_cam_stats.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "grad_cam_near_parasite.txt").open("w") as f:
        f.write("\n".join(near_parasite_lines) + "\n")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, c in enumerate(("tp", "tn", "fp", "fn")):
        arr = groups[c]
        if not arr:
            for r in range(2):
                axes[r, i].set_visible(False)
            continue
        xs = [x["peak_x"] for x in arr]
        ys = [x["peak_y"] for x in arr]
        axes[0, i].hist2d(xs, ys, bins=20, range=[[0, 1], [0, 1]], cmap="hot")
        axes[0, i].invert_yaxis()
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(1, 0)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_title(f"{c.upper()}  (n={len(arr)})  peak density")
        axes[0, i].set_xlabel("peak col")
        axes[0, i].set_ylabel("peak row")
        bfs = [x["boundary_fraction"] for x in arr]
        axes[1, i].hist(bfs, bins=20, range=(0, 1), color="steelblue")
        axes[1, i].set_title(
            f"{c.upper()}  boundary-fraction (mean={np.mean(bfs):.2f})"
        )
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_xlabel("mass in outer 10%")
    plt.tight_layout()
    plt.savefig(out_dir / "grad_cam_stats_plot.png", dpi=120)
    plt.close(fig)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote:")
    print(f"  {out_dir / 'grad_cam_stats.json'}")
    print(f"  {out_dir / 'grad_cam_stats_plot.png'}")
    print(f"  {out_dir / 'grad_cam_near_parasite.txt'}")


if __name__ == "__main__":
    main()
