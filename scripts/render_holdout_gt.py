#!/usr/bin/env python3
"""
Render ground-truth (GT) segmentation overlays for the blind holdout set.

Outputs PNGs with real masks blended on the original RGB image size.
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

from steel_defect.config import load_yaml, set_seed  # noqa: E402
from steel_defect.dataset import build_image_index, load_holdout_ids  # noqa: E402
from steel_defect.overlay import overlay_masks  # noqa: E402
from steel_defect.rle import masks_to_multichannel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render_gt_holdout")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "improved.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "overlays" / "gt_holdout")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

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
            f"Holdout id mismatch: df has {len(holdout_df)} rows but ids set has {len(holdout_ids)}. "
            "Check that all ids exist in train.csv/images_dir."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    saved = []

    for _, row in holdout_df.iterrows():
        img_path = Path(row["path"])
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        rles = {
            1: row["rle_1"],
            2: row["rle_2"],
            3: row["rle_3"],
            4: row["rle_4"],
        }
        # (C, H, W) with 0/1 float values.
        gt_masks = masks_to_multichannel(rles, h, w, num_classes=int(cfg["model"].get("num_classes", 4)))

        over_rgb = overlay_masks(rgb, gt_masks, threshold=args.threshold, alpha=args.alpha)
        out_path = args.out / f"{str(row['ImageId'])}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(over_rgb, cv2.COLOR_RGB2BGR))
        saved.append(str(out_path))

    meta = {
        "config": str(args.config),
        "n_holdout": int(len(holdout_df)),
        "out_dir": str(args.out),
        "threshold": args.threshold,
        "alpha": args.alpha,
        "image_ids": [str(x) for x in holdout_df["ImageId"].tolist()],
    }
    (args.out / "render_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("saved %s overlays → %s", len(saved), args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

