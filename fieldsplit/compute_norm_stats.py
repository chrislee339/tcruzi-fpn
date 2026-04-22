import os
from pathlib import Path
import numpy as np
from PIL import Image

TRAIN_DIR = Path("/home/chris/Documents/chagas/dataset_split_fieldlevel/train")
OUT = Path("/home/chris/Documents/chagas/fieldsplit/norm_stats.txt")


def iter_images():
    for cls in ("positive_images", "negative_images"):
        d = TRAIN_DIR / cls
        for fn in os.listdir(d):
            yield (d / fn)


def main() -> None:
    n_pixels = 0
    sum_c = np.zeros(3, dtype=np.float64)
    sq_c = np.zeros(3, dtype=np.float64)
    for i, p in enumerate(iter_images(), start=1):
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        n_pixels += h * w
        sum_c += arr.reshape(-1, 3).sum(axis=0)
        sq_c += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        if i % 500 == 0:
            print(f"  processed {i} images...")
    mean = sum_c / n_pixels
    var = sq_c / n_pixels - mean**2
    std = np.sqrt(np.clip(var, 0.0, None))
    line_mean = "[" + ", ".join((f"{m:.4f}" for m in mean)) + "]"
    line_std = "[" + ", ".join((f"{s:.4f}" for s in std)) + "]"
    print()
    print(f"mean = {line_mean}")
    print(f"std  = {line_std}")
    with OUT.open("w") as f:
        f.write(f"n_images   = {i}\n")
        f.write(f"n_pixels   = {n_pixels}\n")
        f.write(f"mean       = {line_mean}\n")
        f.write(f"std        = {line_std}\n")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
