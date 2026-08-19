"""CLI: predict on a single image and optionally save overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from steel_defect.inference import load_predictor
from steel_defect.overlay import overlay_masks


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Steel defect segmentation CLI")
    p.add_argument("image", type=Path)
    p.add_argument("--weights", type=Path, default=Path("artifacts/model.pt"))
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    p.add_argument("--out", type=Path, default=Path("artifacts/overlay.png"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args(argv)

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")

    weights = str(args.weights) if args.weights.exists() else None
    onnx = str(args.onnx) if args.onnx and args.onnx.exists() else None
    pred = load_predictor(
        backend=args.backend,
        weights=weights,
        onnx_path=onnx,
        image_size=args.image_size,
    )
    probs = pred.predict(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    over = overlay_masks(rgb, probs, threshold=args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    for i in range(probs.shape[0]):
        area = float((probs[i] >= args.threshold).mean())
        print(f"class {i + 1}: positive_area_frac={area:.4f} max_prob={probs[i].max():.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
