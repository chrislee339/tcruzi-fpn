from pathlib import Path
from PIL import Image

SRC = Path("/home/chris/Documents/chagas/paper_figures")
OUT = SRC / "plos_submission"
OUT.mkdir(parents=True, exist_ok=True)
MAPPING = [
    ("fig1_full_field_annotated.png", "Fig1.tif"),
    ("fig2_quadrant_split.png", "Fig2.tif"),
    ("fig3_white_balance.png", "Fig3.tif"),
    ("fig4_flip_augmentation.png", "Fig4.tif"),
    ("fig_test_confmat_roc_fieldsplit.png", "Fig6_Fig7.tif"),
    ("fig_prob_histogram_fieldsplit.png", "Fig8.tif"),
    ("fig_threshold_sweep_fieldsplit.png", "Fig9.tif"),
    ("fig_gradcam_grid_fieldsplit.png", "Fig10.tif"),
    ("fig_gradcam_tp.png", "S1_Fig.tif"),
    ("fig_gradcam_tn.png", "S2_Fig.tif"),
    ("fig_gradcam_fp.png", "S3_Fig.tif"),
    ("fig_gradcam_fn.png", "S4_Fig.tif"),
]


def convert(src: Path, dst: Path, dpi: int = 300):
    im = Image.open(src).convert("RGB")
    im.save(dst, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    return im.size


def main() -> None:
    print(f"Converting figures to {OUT}/ at 300 DPI TIFF…")
    missing = []
    wrote = 0
    for src_name, dst_name in MAPPING:
        src = SRC / src_name
        dst = OUT / dst_name
        if not src.exists():
            missing.append(src_name)
            continue
        size = convert(src, dst)
        kb = dst.stat().st_size / 1024
        print(f"  {src_name:46s} -> {dst_name:16s}  {size[0]}x{size[1]} ({kb:.0f} KB)")
        wrote += 1
    print(f"\nWrote {wrote} TIFFs to {OUT}/")
    if missing:
        print(
            f"\nNote — {len(missing)} expected source(s) missing (create then rerun):"
        )
        for m in missing:
            print(f"  {m}")
    print("\nStill TO DO before upload:")
    print(
        "  * Fig 5 (architecture diagram) — create manually (e.g. in Inkscape or draw.io),"
    )
    print(
        "    export as EPS/600 DPI or TIFF/600 DPI, save to paper_figures/plos_submission/Fig5.tif"
    )
    print(
        "  * Split Fig6_Fig7.tif (currently the combined confusion-matrix + ROC panel)"
    )
    print(
        "    into two separate TIFFs if the manuscript lists Fig6 and Fig7 independently,"
    )
    print("    or update the Figure Legends to reference a single combined panel.")


if __name__ == "__main__":
    main()
