# Interview script — P01 Steel Defect Segmentation

## 30-second pitch

I built a Severstal-style **pixel-level defect segmentation** demo: multi-label UNet, explicit **class imbalance** handling (focal + class weights), per-class Dice, FastAPI overlay, and ONNX CPU latency. Hire signal is not “I can call SMP” — it is **baseline vs improved metrics with an imbalance story**.

## Likely questions → answers

### Why multi-label (4 channels) instead of softmax?

Severstal annotations allow **multiple defect classes on one plate**. Softmax forces mutual exclusivity; independent sigmoid heads match the label schema and RLE-per-class CSV.

### How did you handle imbalance?

Most pixels are background; class 2 is typically rare. Baseline uses Dice + BCE with uniform weights. Improved uses **focal BCE (γ=2)** plus **higher pos weights on rare classes**, keeping Dice for region overlap. Report **per-class Dice**, not only mean.

### Validation / leakage?

Split by **ImageId** stratified on defect presence. **19 blind holdout** ImageIds (`data/holdout_ids.txt`) excluded from train and val. Same seed in YAML for train/eval/holdout scripts.

### Why two Dice numbers?

Default per-class Dice averages over all images; empty GT + empty pred → Dice = 1.0. Report **`per_class_dice_present_only`** from `eval_holdout.py`. Holdout: improved class 3 present-only Dice **0.20** vs baseline **0.05**.

### Why UNet-ResNet34?

Strong industrial baseline: ImageNet encoder, cheap enough for 256² demos, easy ONNX export. FPN is a config switch for a one-line ablation.

### Metric choice?

Show val table + **blind holdout present-only** table. Mean Dice alone hides class head collapse (e.g. class 4 val Dice ~0.94 but holdout class 4 recall = 0).

### ONNX / serving?

Train in PyTorch (CUDA when available). Export ONNX; serve with Torch or ORT via `INFERENCE_BACKEND`. API returns class areas + **PNG overlay** for visual QA.

### Failure modes?

- **Multi-class collapse:** detects defects as class 3; class 4 (yellow GT) not predicted on holdout  
- Thin scratches vs mill texture → false positives  
- Resize/pad to 256² changes thin defect geometry  

### Production gap?

No mill private data, no TensorRT required, no Kaggle leaderboard farming. Next: plate-level grouping, stronger augs, calibration of thresholds per class.

## Demo commands to memorize

```bash
python scripts/download_data.py --source synthetic --subset 64
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/improved.yaml
python scripts/export_onnx.py --config configs/improved.yaml
docker compose -f docker/docker-compose.yml up --build
curl -s http://127.0.0.1:8000/health
```
