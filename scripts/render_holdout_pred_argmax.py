#!/usr/bin/env python3
"""
Render argmax prediction overlays for the blind holdout set.

For each pixel: class = argmax_c(prob[c]).
Optionally only color pixels where max_prob >= min_prob.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect.config import load_yaml, resolve_device, set_seed  # noqa: E402
from steel_defect.dataset import build_image_index, load_holdout_ids  # noqa: E402
from steel_defect.inference import load_predictor  # noqa: E402
from steel_defect.overlay import CLASS_COLORS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render_argmax_holdout")


def overlay_argmax(
    image_rgb: np.ndarray,
    probs: np.ndarray,
    min_prob: float = 0.3,
    alpha: float = 0.45,
) -> np.ndarray:
    """image_rgb: uint8 (H,W,3), probs: float32 (C,H,W)."""
    if image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb must be uint8")
    if probs.ndim != 3:
        raise ValueError(f"probs must be (C,H,W), got {probs.shape}")

    h, w = image_rgb.shape[:2]
    if probs.shape[1:] != (h, w):
        raise ValueError(f"probs spatial {probs.shape[1:]} != image {(h,w)}")

    argmax_c = probs.argmax(axis=0)  # (H,W)
    max_prob = probs.max(axis=0)  # (H,W)
    mask_col = max_prob >= min_prob

    out = image_rgb.astype(np.float32).copy()
    c = probs.shape[0]
    for ci in range(c):
        bin_m = mask_col & (argmax_c == ci)
        if not bin_m.any():
            continue
        color = CLASS_COLORS[ci % len(CLASS_COLORS)].astype(np.float32)
        out[bin_m] = (1 - alpha) * out[bin_m] + alpha * color

    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "improved.yaml")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "overlays" / "argmax_holdout")
    ap.add_argument("--min-prob", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--threshold-show", type=float, default=0.3, help="used only for logging (max prob)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "cuda")))

    data_root = ROOT / cfg["data"]["root"]
    index = build_image_index(data_root / cfg["data"]["csv"], data_root / cfg["data"]["images_dir"])
    holdout_path = cfg["data"].get("holdout_ids")
    if not holdout_path:
        raise SystemExit("cfg.data.holdout_ids is missing. Add data/holdout_ids.txt path to config.")
    holdout_ids = load_holdout_ids(ROOT / holdout_path)
    if not holdout_ids:
        raise SystemExit("Holdout ids empty. data/holdout_ids.txt is empty?")

    holdout_df = index[index["ImageId"].astype(str).isin(holdout_ids)].reset_index(drop=True)
    if len(holdout_df) != len(holdout_ids):
        raise SystemExit(
            f"Holdout id mismatch: df has {len(holdout_df)} rows but ids set has {len(holdout_ids)}."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor(
        backend="torch",
        weights=str(args.weights),
        onnx_path=None,
        device=device,
        image_size=int(cfg["data"]["image_size"]),
    )

    saved = []
    per_image_log = []
    argmax_all_counts: np.ndarray | None = None
    argmax_colored_counts: np.ndarray | None = None
    for _, row in holdout_df.iterrows():
        img_path = Path(row["path"])
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        probs = predictor.predict(bgr)  # (C,H,W) on original size

        argmax_c = probs.argmax(axis=0)  # (H,W)
        max_prob = probs.max(axis=0)  # (H,W)
        mask_col = max_prob >= args.min_prob

        c = probs.shape[0]
        if argmax_all_counts is None:
            argmax_all_counts = np.zeros(c, dtype=np.float64)
            argmax_colored_counts = np.zeros(c, dtype=np.float64)
        for ci in range(c):
            argmax_all_counts[ci] += float((argmax_c == ci).sum())
            argmax_colored_counts[ci] += float((mask_col & (argmax_c == ci)).sum())

        over = overlay_argmax(rgb, probs, min_prob=args.min_prob, alpha=args.alpha)

        out_path = args.out / f"{str(row['ImageId'])}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        saved.append(str(out_path))

        per_image_log.append(
            {
                "image_id": str(row["ImageId"]),
                "max_prob": float(max_prob.max()),
                "argmax_counts": [int((argmax_c == ci).sum()) for ci in range(c)],
                "argmax_colored_counts": [int((mask_col & (argmax_c == ci)).sum()) for ci in range(c)],
            }
        )

    meta = {
        "config": str(args.config),
        "weights": str(args.weights),
        "split": "blind_holdout",
        "n_holdout": len(holdout_df),
        "out_dir": str(args.out),
        "min_prob": args.min_prob,
        "alpha": args.alpha,
        "max_prob_stats": {
            "max": float(np.max([x["max_prob"] for x in per_image_log])) if per_image_log else None,
            "mean": float(np.mean([x["max_prob"] for x in per_image_log])) if per_image_log else None,
        },
        "argmax_pixel_fraction_all": None,
        "argmax_pixel_fraction_colored": None,
        "per_image": per_image_log,
    }

    if argmax_all_counts is not None and argmax_colored_counts is not None:
        total_all = float(argmax_all_counts.sum())
        total_colored = float(argmax_colored_counts.sum())
        meta["argmax_pixel_fraction_all"] = (
            (argmax_all_counts / total_all).tolist() if total_all > 0 else None
        )
        meta["argmax_pixel_fraction_colored"] = (
            (argmax_colored_counts / total_colored).tolist() if total_colored > 0 else None
        )

    (args.out / "render_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("saved %s argmax overlays → %s", len(saved), args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

