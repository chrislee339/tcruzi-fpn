# tcruzi-fpn

## Overview

This repository contains the training and evaluation code for a custom feature pyramid network (FPN) that classifies smartphone-acquired blood-smear images for *T. cruzi* presence, together with the blinded re-audit web tool used to quantify annotation reliability on the underlying dataset. The paper reports AUC 0.961 on a test set held out at the slide level, with Grad-CAM evidence that the model attends to parasite morphology and a three-epidemiologist re-audit finding that 9 of the 44 model/label disagreements are in fact annotation errors in the source dataset.

## Repository layout

```
tcruzi-fpn/
├── environment.yml                 # conda environment (PyTorch + CUDA 11.3)
├── train_fpn_v2.py                 # standalone training script (original paper setup)
├── fieldsplit/                     # slide-level training + evaluation pipeline
│   ├── make_fieldsplit.py          # build slide-level train/val/test directories
│   ├── compute_norm_stats.py       # dataset mean/std for normalization
│   ├── train_fieldsplit.py         # main training loop
│   ├── train_hardened.py           # deterministic single-GPU training recipe
│   ├── train_ablation.py           # architectural ablation (backbone / fpn_p5 / fpn_multi)
│   ├── eval_fieldsplit.py          # test-set evaluation
│   ├── grad_cam_fieldsplit.py      # Grad-CAM overlays
│   ├── grad_cam_analysis.py        # quantitative CAM-to-parasite localization
│   ├── build_audit_package.py     # assemble blinded audit images + scoring template
│   ├── analyze_reader_scores.py    # aggregate per-reader scores into consensus
│   ├── make_paper_figures_1to4.py  # Figs 1–4 (dataset + preprocessing)
│   ├── make_extra_figures.py       # Figs 8–9 (probability distribution, threshold sweep)
│   ├── make_fig5_drawio.py         # Fig 5 (architecture diagram, draw.io XML)
│   └── convert_figures_for_plos.py # PNG → 300 DPI TIFF for submission
├── chagas-audit/                   # blinded re-audit web tool
│   ├── index.html
│   ├── images/                     # reader-facing renamed images (img_001.png …)
│   └── README.md
└── audit/
    ├── scores/                     # per-reader and consensus score JSONs
    └── S1_Table.xlsx               # 9 overturned image IDs (panel consensus)
```

## Data

The underlying smartphone-microscopy images and parasite-coordinate annotations are from Morais et al. (2022) and are distributed through the Supplemental Information of that publication. The preprocessed 1,224 × 1,632 quadrant images used as direct inputs to our classifier (~16 GB, 5,788 images across train/val/test) are deposited at Zenodo: **doi:10.5281/zenodo.19711048**.

## Environment

Built for Linux with CUDA 11.3. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tcruzi-fpn
```

## Quick start

Build the slide-level train/val/test splits (requires the Morais dataset plus preprocessing):

```bash
python fieldsplit/make_fieldsplit.py
```

Train the headline model (hardened deterministic single-GPU recipe, seed 69):

```bash
python fieldsplit/train_hardened.py --seed 69
```

Evaluate on the held-out test set:

```bash
python fieldsplit/eval_fieldsplit.py --checkpoint checkpoints/hardened_seed69/best_model_checkpoint.pth
```

Regenerate Grad-CAM overlays and quantitative localization stats:

```bash
python fieldsplit/grad_cam_fieldsplit.py
python fieldsplit/grad_cam_analysis.py
```

## License

MIT License. See `LICENSE`.

## Citation

If you use this code, please cite the archived release:

> Lee C, Gual-Gonzalez L, Lynn MK, Braumuller K, Nolan MS, Valafar H. *tcruzi-fpn: code and audit tooling for a feature pyramid classifier of* Trypanosoma cruzi *in smartphone blood smear microscopy.* Zenodo. 2026. doi:10.5281/zenodo.19711401

and the accompanying dataset:

> *tcruzi-fpn dataset: slide-level partitioned quadrant images for* Trypanosoma cruzi *classification from the Morais et al. smartphone microscopy corpus.* Zenodo. 2026. doi:10.5281/zenodo.19711048
