import os
import re
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path("/home/chris/Documents/chagas/dataset_split")
DST = Path("/home/chris/Documents/chagas/dataset_split_fieldlevel")
TRAIN_MAX_FIELD = 519
VAL_MAX_FIELD = 599
SEED = 42
NOISE_STD = 50.0
ORIG_QUAD_RE = re.compile("^field\\d+_quad\\d+\\.png$")
FIELD_NUM_RE = re.compile("^field(\\d+)")


def is_original_quad(fn: str) -> bool:
    return bool(ORIG_QUAD_RE.fullmatch(fn))


def field_num(fn: str) -> int:
    m = FIELD_NUM_RE.match(fn)
    return int(m.group(1)) if m else -1


def collect_originals() -> dict[str, str]:
    quads: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for cls in ("positive_images", "negative_images"):
            d = SRC / split / cls
            if not d.is_dir():
                continue
            for fn in os.listdir(d):
                if is_original_quad(fn):
                    prev = quads.get(fn)
                    if prev is not None and prev != cls:
                        raise RuntimeError(
                            f"class conflict for {fn}: {prev!r} vs {cls!r}"
                        )
                    quads[fn] = cls
    return quads


def partition_by_field(quads: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    out = {"train": [], "val": [], "test": []}
    for fn, cls in quads.items():
        n = field_num(fn)
        if n <= TRAIN_MAX_FIELD:
            out["train"].append((fn, cls))
        elif n <= VAL_MAX_FIELD:
            out["val"].append((fn, cls))
        else:
            out["test"].append((fn, cls))
    return out


def find_src_path(fn: str, cls: str) -> Path:
    for split in ("train", "val", "test"):
        p = SRC / split / cls / fn
        if p.exists():
            return p
    raise FileNotFoundError(f"no source found for {fn} in {cls}")


def copy_originals(parts: dict[str, list[tuple[str, str]]]) -> None:
    for split, items in parts.items():
        for fn, cls in items:
            dst = DST / split / cls / fn
            if not dst.exists():
                shutil.copy2(find_src_path(fn, cls), dst)


def hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def balance_train_with_flips(rng: random.Random) -> None:
    tp = DST / "train" / "positive_images"
    tn = DST / "train" / "negative_images"
    pos = sorted(os.listdir(tp))
    neg = sorted(os.listdir(tn))
    if len(pos) <= len(neg):
        minority_dir, minority, majority_n = (tp, pos, len(neg))
    else:
        minority_dir, minority, majority_n = (tn, neg, len(pos))
    flip_kinds = ("hflip", "vflip", "hvflip")
    jobs: list[tuple[str, str]] = []
    for k in flip_kinds:
        for fn in minority:
            jobs.append((fn, k))
    rng.shuffle(jobs)
    added = 0
    i = 0
    while len(minority) + added < majority_n and i < len(jobs):
        fn, kind = jobs[i]
        i += 1
        out_name = fn.replace(".png", f"_{kind}.png")
        out_path = minority_dir / out_name
        if out_path.exists():
            continue
        with Image.open(minority_dir / fn) as im:
            im = im.convert("RGB")
            if kind == "hflip":
                im2 = hflip(im)
            elif kind == "vflip":
                im2 = vflip(im)
            else:
                im2 = hflip(vflip(im))
            im2.save(out_path, format="PNG", optimize=False)
        added += 1


def add_gaussian_noise(
    img: Image.Image, std: float, rng: np.random.Generator
) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0.0, std, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def noise_augment_train(np_rng: np.random.Generator) -> None:
    for cls in ("positive_images", "negative_images"):
        d = DST / "train" / cls
        existing = sorted(os.listdir(d))
        for fn in existing:
            if fn.startswith("noisy_"):
                continue
            out_name = f"noisy_{fn}"
            out_path = d / out_name
            if out_path.exists():
                continue
            with Image.open(d / fn) as im:
                noisy = add_gaussian_noise(im, NOISE_STD, np_rng)
                noisy.save(out_path, format="PNG", optimize=False)


def main() -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    print(f"Scanning {SRC} for original quadrant images...")
    quads = collect_originals()
    pos = sum((1 for c in quads.values() if c == "positive_images"))
    neg = sum((1 for c in quads.values() if c == "negative_images"))
    print(f"  found {len(quads)} originals: {pos} positive, {neg} negative")
    parts = partition_by_field(quads)
    print("\nField-number partition:")
    for s in ("train", "val", "test"):
        p = sum((1 for _, c in parts[s] if c == "positive_images"))
        n = sum((1 for _, c in parts[s] if c == "negative_images"))
        print(f"  {s:5s}: {p:4d} pos + {n:4d} neg = {p + n:4d} quadrants")
    print("\nCreating output directories...")
    for s in ("train", "val", "test"):
        for cls in ("positive_images", "negative_images"):
            (DST / s / cls).mkdir(parents=True, exist_ok=True)
    print("Copying originals...")
    copy_originals(parts)
    print("Flip-augmenting the training minority class...")
    balance_train_with_flips(rng)
    print("Applying Gaussian-noise augmentation to training set...")
    noise_augment_train(np_rng)
    print("\nFinal split counts:")
    for s in ("train", "val", "test"):
        for cls in ("positive_images", "negative_images"):
            n = len(os.listdir(DST / s / cls))
            print(f"  {s:5s}/{cls:15s}: {n}")
    manifest = DST / "manifest.txt"
    with manifest.open("w") as f:
        f.write("Field-level (≈ slide-level) split, Morais-compatible.\n")
        f.write(f"seed: {SEED}\n")
        f.write(f"train: fields 0001 – {TRAIN_MAX_FIELD:04d}\n")
        f.write(f"val  : fields {TRAIN_MAX_FIELD + 1:04d} – {VAL_MAX_FIELD:04d}\n")
        f.write(f"test : fields {VAL_MAX_FIELD + 1:04d} – 0704 (Morais test block)\n")
        f.write(
            f"augmentation (train only): flip-balance on minority class, Gaussian noise (std={NOISE_STD}) on every training image.\n"
        )
        f.write(f"val/test contain originals only.\n")
    print(f"\nManifest written to {manifest}")


if __name__ == "__main__":
    main()
