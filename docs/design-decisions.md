# Design decisions — P01 Steel Defect Segmentation

## Problem framing

Industrial steel defect segmentation is a **multi-label, heavily imbalanced** dense prediction task. The portfolio goal is to show: (1) correct label geometry (RLE ↔ mask), (2) a honest baseline, (3) an imbalance-aware improvement with **per-class Dice**, (4) deployable inference (FastAPI + ONNX).

## Dataset & license

- Primary: Kaggle `severstal-steel-defect-detection` (competition rules / research).  
- Smoke path: synthetic generator or stratified 500-image subset.  
- Raw images never committed; `data/README.md` documents terms.

## Validation

- Image-level split, stratified by `has_defect`, `val_frac=0.2`, `seed=42`.  
- **Blind holdout:** 19 ImageIds in `data/holdout_ids.txt` carved out before train/val — never used for checkpoint selection.  
- Metrics: report both mean Dice and **`present_only`** Dice (only images where GT contains that class).

## Known result (Severstal full train)

| | Val best Dice | Holdout Dice | Holdout class 3 Dice (present only) |
|---|--:|--:|--:|
| Baseline | 0.860 | 0.687 | 0.046 |
| Improved | 0.867 | 0.737 | 0.201 |

Improved wins on holdout overall and on class 3 where defects exist. **Class 4 head collapse** remains — val Dice for class 4 is misleading (mostly empty-empty matches).

## Architecture

| Choice | Why |
|--------|-----|
| `segmentation-models-pytorch` UNet + ResNet34 | Standard industrial baseline; pretrained encoder; simple ONNX |
| 4 sigmoid heads | Multi-class co-occurrence on one image |
| 256 input | Speed/memory on RTX 2070-class GPUs; document accuracy/speed trade-off |
| Optional FPN via config | Ablation without new code paths |

## Losses

| Mode | Recipe |
|------|--------|
| `baseline` | Soft Dice + BCEWithLogits (uniform class weights) |
| `improved` | Soft Dice + **focal BCE** + **class weights** (boost rare class 2) |

Focal down-weights easy background pixels; Dice stabilizes overlap for sparse positives.

## Training defaults
- Live `tqdm` + logging of loss / ETA / GPU mem.  
- Best checkpoint by val mean Dice.

## Serving

- FastAPI: `/health`, `/predict` with optional base64 overlay.  
- Docker CPU torch/ORT for portability; local train uses CUDA wheels.  
- Weights mounted at runtime — not baked into the image.

## What we explicitly skipped

- Kaggle submission farming  
- Private mill data  
- TensorRT (optional later if GPU deploy matters)  
- Full-resolution 1600×256 training in CI  

## Accuracy vs speed

Higher `image_size` and longer training improve thin-defect Dice but increase ONNX CPU latency (documented in `artifacts/onnx_latency.json` after export). Portfolio default favors a reproducible 256² demo with clear metric tables.
