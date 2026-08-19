#!/usr/bin/env python3
"""
Report which holdout images predict a given class at a fixed threshold.

For multi-label segmentation: positive_area_frac = mean(probs[c] >= threshold).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect.config import load_yaml, resolve_device, set_seed  # noqa: E402
from steel_defect.dataset import build_image_index, load_holdout_ids  # noqa: E402
from steel_defect.inference import load_predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("holdout_report")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "improved.yaml")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--class-id", type=int, default=4, help="1..4 (Severstal classes)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "cuda")))

    data_root = ROOT / cfg["data"]["root"]
    index = build_image_index(data_root / cfg["data"]["csv"], data_root / cfg["data"]["images_dir"])
    holdout_path = cfg["data"].get("holdout_ids")
    if not holdout_path:
        raise SystemExit("cfg.data.holdout_ids is missing (data/holdout_ids.txt).")
    holdout_ids = load_holdout_ids(ROOT / holdout_path)
    if not holdout_ids:
        raise SystemExit("Holdout ids empty.")

    holdout_df = index[index["ImageId"].astype(str).isin(holdout_ids)].reset_index(drop=True)
    if len(holdout_df) != len(holdout_ids):
        log.warning(
            "holdout_df rows (%s) != holdout_ids size (%s)", len(holdout_df), len(holdout_ids)
        )

    class_id = int(args.class_id)
    if class_id < 1 or class_id > 4:
        raise SystemExit("--class-id must be between 1 and 4")
    c_idx = class_id - 1

    predictor = load_predictor(
        backend="torch",
        weights=str(args.weights),
        onnx_path=None,
        device=device,
        image_size=int(cfg["data"]["image_size"]),
    )

    positive = []
    per_image = []
    for _, row in holdout_df.iterrows():
        img_path = Path(row["path"])
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(str(img_path))
        probs = predictor.predict(bgr)  # (C,H,W)
        pos_area = float((probs[c_idx] >= args.threshold).mean())
        mx = float(probs[c_idx].max())
        per_image.append(
            {
                "image_id": str(row["ImageId"]),
                "positive_area_frac": pos_area,
                "max_prob": mx,
            }
        )
        if pos_area > 0:
            positive.append(
                {"image_id": str(row["ImageId"]), "positive_area_frac": pos_area, "max_prob": mx}
            )

    out = args.out or (ROOT / "artifacts" / "holdout_class4_report.json")
    out_payload = {
        "config": str(args.config),
        "weights": str(args.weights),
        "threshold": args.threshold,
        "class_id": class_id,
        "n_holdout": len(holdout_df),
        "n_positive": len(positive),
        "positive_image_ids": [x["image_id"] for x in positive],
        "per_image": per_image,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    n_pos = len(positive)
    n_total = len(holdout_df)
    print(f"class {class_id} @ threshold {args.threshold}: n_positive={n_pos}/{n_total}")
    print("positive_image_ids:", ", ".join([x["image_id"] for x in positive]))


if __name__ == "__main__":
    main()
