import csv
import re
from collections import defaultdict
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None
OUT = Path("/home/chris/Documents/chagas/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
QUAD_ROOT = Path("/home/chris/Documents/chagas/dataset_split_fieldlevel")
MORAIS_CSV = Path("/home/chris/Documents/chagas/peerj-10-13470-s001_all.csv")


def load_annotations() -> dict[str, list[tuple[float, float, str]]]:
    per_field: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with MORAIS_CSV.open() as f:
        for r in csv.DictReader(f):
            fn = r["filename"].replace(".jpg", "")
            per_field[fn].append((float(r["x"]), float(r["y"]), r["structure"]))
    return per_field


def find_quadrant_file(field: str, quad: int) -> Path | None:
    for split in ("train", "val", "test"):
        for cls in ("positive_images", "negative_images"):
            p = QUAD_ROOT / split / cls / f"{field}_quad{quad}.png"
            if p.exists():
                return p
    return None


def stitch_quadrants(field: str) -> Image.Image | None:
    tiles = [find_quadrant_file(field, q) for q in (1, 2, 3, 4)]
    if not all(tiles):
        return None
    ims = [Image.open(t).convert("RGB") for t in tiles]
    W, H = ims[0].size
    full = Image.new("RGB", (2 * W, 2 * H))
    full.paste(ims[0], (0, 0))
    full.paste(ims[1], (W, 0))
    full.paste(ims[2], (0, H))
    full.paste(ims[3], (W, H))
    return full


def scale_coord(x: float, y: float, orig_w: int, orig_h: int, new_w: int, new_h: int):
    return (x * new_w / orig_w, y * new_h / orig_h)


def infer_orig_size(anns: list[tuple[float, float, str]]) -> tuple[int, int]:
    max_x = max((a[0] for a in anns)) if anns else 0
    max_y = max((a[1] for a in anns)) if anns else 0
    return (3456, 4608) if max_x > 2448 or max_y > 3264 else (2448, 3264)


def make_fig1(field="field0668", anns_by_field=None):
    ann = anns_by_field[field]
    stitched = stitch_quadrants(field)
    if stitched is None:
        raise RuntimeError(f"Cannot stitch {field} — quadrants missing")
    sw, sh = stitched.size
    ow, oh = infer_orig_size(ann)
    nuc = [(a[0], a[1]) for a in ann if a[2] == "NUCLEUS"]
    kin = [(a[0], a[1]) for a in ann if a[2] == "KINETOPLAST"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=300)
    for ax, overlay in zip(axes, (False, True)):
        ax.imshow(stitched)
        ax.set_xticks([])
        ax.set_yticks([])
        if overlay:
            for x, y in nuc:
                xs, ys = scale_coord(x, y, ow, oh, sw, sh)
                ax.plot(
                    xs,
                    ys,
                    marker="+",
                    markersize=14,
                    markeredgewidth=2,
                    color="#ff3b3b",
                    linestyle="",
                )
            for x, y in kin:
                xs, ys = scale_coord(x, y, ow, oh, sw, sh)
                ax.plot(
                    xs,
                    ys,
                    marker="+",
                    markersize=14,
                    markeredgewidth=2,
                    color="#35c35c",
                    linestyle="",
                )
    axes[0].set_title("Raw smartphone-microscopy capture", fontsize=11)
    axes[1].set_title(
        "With Morais parasite annotations\n(red = nucleus, green = kinetoplast)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig1_full_field_annotated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig2(field="field0668", anns_by_field=None):
    ann = anns_by_field[field]
    stitched = stitch_quadrants(field)
    sw, sh = stitched.size
    ow, oh = infer_orig_size(ann)
    centroids = []
    i = 0
    pts = [a for a in ann]
    while i < len(pts):
        s1, x1, y1 = (pts[i][2], pts[i][0], pts[i][1])
        if i + 1 < len(pts):
            s2, x2, y2 = (pts[i + 1][2], pts[i + 1][0], pts[i + 1][1])
            if {s1, s2} == {"NUCLEUS", "KINETOPLAST"}:
                centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
                i += 2
                continue
        centroids.append((x1, y1))
        i += 1
    QW, QH = (ow // 2, oh // 2)
    quad_has_parasite = {1: False, 2: False, 3: False, 4: False}
    for cx, cy in centroids:
        if cx < QW and cy < QH:
            q = 1
        elif cx >= QW and cy < QH:
            q = 2
        elif cx < QW and cy >= QH:
            q = 3
        else:
            q = 4
        quad_has_parasite[q] = True
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    ax.imshow(stitched)
    ax.axvline(sw / 2, color="#ffbf00", lw=2, linestyle="--")
    ax.axhline(sh / 2, color="#ffbf00", lw=2, linestyle="--")
    label_pos = {
        1: (sw / 4, sh / 4),
        2: (3 * sw / 4, sh / 4),
        3: (sw / 4, 3 * sh / 4),
        4: (3 * sw / 4, 3 * sh / 4),
    }
    for q, (px, py) in label_pos.items():
        label = "POSITIVE" if quad_has_parasite[q] else "NEGATIVE"
        color = "#35c35c" if quad_has_parasite[q] else "#ff3b3b"
        ax.text(
            px,
            py,
            f"Quad {q}\n{label}",
            fontsize=13,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor=color, edgecolor="none", alpha=0.9
            ),
        )
    for cx, cy in centroids:
        xs, ys = scale_coord(cx, cy, ow, oh, sw, sh)
        ax.plot(
            xs,
            ys,
            marker="+",
            markersize=16,
            markeredgewidth=2.5,
            color="cyan",
            linestyle="",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Quadrant-based labeling: each field is split into four sub-images;\na quadrant is POSITIVE if an annotated parasite centroid (cyan +) falls within it.",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig2_quadrant_split.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3(quad_path=None):
    if quad_path is None:
        quad_path = find_quadrant_file("field0650", 2) or find_quadrant_file(
            "field0100", 1
        )
    if quad_path is None:
        raise RuntimeError("no quadrant found for fig 3")
    img = np.asarray(Image.open(quad_path).convert("RGB"), dtype=np.float32)
    cast = img.copy()
    cast[..., 0] = np.clip(cast[..., 0] * 1.12 + 10, 0, 255)
    cast[..., 2] = np.clip(cast[..., 2] * 0.86 - 5, 0, 255)
    cast = cast.astype(np.uint8)
    if cv2 is not None and hasattr(cv2, "xphoto"):
        bgr = cv2.cvtColor(cast, cv2.COLOR_RGB2BGR)
        wb = cv2.xphoto.createSimpleWB()
        balanced = cv2.cvtColor(wb.balanceWhite(bgr), cv2.COLOR_BGR2RGB)
    else:
        mean = cast.mean(axis=(0, 1))
        gain = mean.mean() / (mean + 1e-06)
        balanced = np.clip(cast.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), dpi=300)
    axes[0].imshow(cast)
    axes[0].set_title("Before white-balance correction", fontsize=11)
    axes[1].imshow(balanced)
    axes[1].set_title("After white-balance correction", fontsize=11)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / "fig3_white_balance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig4(quad_path=None):
    if quad_path is None:
        for split in ("train", "val", "test"):
            d = QUAD_ROOT / split / "negative_images"
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.suffix == ".png" and (not f.name.startswith("noisy_")):
                        quad_path = f
                        break
            if quad_path:
                break
    if quad_path is None:
        raise RuntimeError("no negative quadrant found for fig 4")
    im = Image.open(quad_path).convert("RGB")
    hflip = im.transpose(Image.FLIP_LEFT_RIGHT)
    vflip = im.transpose(Image.FLIP_TOP_BOTTOM)
    both = hflip.transpose(Image.FLIP_TOP_BOTTOM)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), dpi=300)
    titles = [
        "Original",
        "Horizontal flip",
        "Vertical flip",
        "Horizontal + vertical flip",
    ]
    imgs = [im, hflip, vflip, both]
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(
        "Flip-based augmentation of a negative-class quadrant to balance the training set",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig4_flip_augmentation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pick_fig1_field(anns_by_field) -> str:
    candidates = []
    for f, anns in anns_by_field.items():
        m = re.match("field(\\d+)", f)
        if not m:
            continue
        n = int(m.group(1))
        if not 600 <= n <= 704:
            continue
        nuc = sum((1 for a in anns if a[2] == "NUCLEUS"))
        kin = sum((1 for a in anns if a[2] == "KINETOPLAST"))
        pairs = min(nuc, kin)
        if not 3 <= pairs <= 6:
            continue
        w, h = infer_orig_size(anns)
        if (w, h) == (2448, 3264) and all(
            (find_quadrant_file(f, q) for q in (1, 2, 3, 4))
        ):
            candidates.append(f)
    return candidates[0] if candidates else "field0668"


if __name__ == "__main__":
    anns = load_annotations()
    fig1_field = pick_fig1_field(anns)
    print(
        f"Fig 1/2 using {fig1_field}  (orig size {infer_orig_size(anns[fig1_field])})"
    )
    make_fig1(fig1_field, anns)
    make_fig2(fig1_field, anns)
    make_fig3()
    make_fig4()
    print("Wrote:")
    for f in sorted(OUT.glob("fig[1-4]_*.png")):
        print(f"  {f}")
