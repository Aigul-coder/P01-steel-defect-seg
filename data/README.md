# Data — Severstal Steel Defect Detection

## Source

- **Kaggle:** [severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
- **Hugging Face (optional mirror):** `Voxel51/severstal_steel_defects`

## License / terms

Competition / research use under **Kaggle competition rules**. Do not redistribute the raw dataset from this repo. Document compliance in any public fork:

1. Accept competition rules on Kaggle before download.
2. Keep images and `train.csv` local under `data/raw/` (gitignored).
3. Do not use for commercial mill deployment without checking current terms.

## Layout after download

```text
data/
  raw/                    # gitignored — full Severstal download
    train_images/
    train.csv
  holdout_ids.txt         # 19 blind QA ImageIds (safe to commit)
  holdout_preview/        # gitignored — local copies for overlay QA
  subset/                 # optional smoke subset
  README.md
```

## Commands

```bash
# Full train set (requires Kaggle credentials ~/.kaggle/kaggle.json)
python scripts/download_data.py --source kaggle

# Smoke subset: 500 images stratified by defect presence
python scripts/download_data.py --source kaggle --subset 500

# Synthetic tiny set for offline / CI-like local smoke (no Kaggle)
python scripts/download_data.py --source synthetic --subset 32
```

## Size

Full competition train images ≈ **1–2 GB**. Prefer subset for pipeline smoke; report hire-signal metrics on a held-out split of the full (or large) train set.
