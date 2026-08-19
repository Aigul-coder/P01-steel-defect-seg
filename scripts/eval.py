#!/usr/bin/env python3
"""Evaluate checkpoint: per-class Dice/IoU on held-out split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect.config import load_yaml, resolve_device, set_seed  # noqa: E402
from steel_defect.dataset import (  # noqa: E402
    SteelDefectDataset,
    build_image_index,
    load_holdout_ids,
    prepare_train_val_holdout,
)
from steel_defect.metrics import DiceIoUAccumulator  # noqa: E402
from steel_defect.models import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "improved.yaml")
    ap.add_argument("--weights", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "cuda")))

    data_root = ROOT / cfg["data"]["root"]
    index = build_image_index(data_root / cfg["data"]["csv"], data_root / cfg["data"]["images_dir"])
    holdout_path = cfg["data"].get("holdout_ids")
    holdout_ids = load_holdout_ids(ROOT / holdout_path if holdout_path else None)
    _, val_df, _ = prepare_train_val_holdout(
        index,
        float(cfg["data"]["val_frac"]),
        int(cfg["seed"]),
        holdout_ids=holdout_ids,
    )
    ds = SteelDefectDataset(val_df, image_size=int(cfg["data"]["image_size"]), train=False)
    loader = DataLoader(
        ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0
    )

    weights = Path(args.weights) if args.weights else ROOT / cfg["artifacts"]["weights"]
    model = build_model(
        encoder=cfg["model"]["encoder"],
        encoder_weights=None,
        num_classes=int(cfg["model"]["num_classes"]),
        architecture=cfg["model"].get("architecture", "unet"),
    ).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    num_classes = int(cfg["model"]["num_classes"])
    acc = DiceIoUAccumulator(num_classes, threshold=float(cfg["eval"]["threshold"]))
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval"):
            images = batch["image"].to(device)
            logits = model(images).cpu().numpy()
            acc.update(
                1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0))),
                batch["mask"].numpy(),
            )
    metrics = acc.result()
    out = {
        "experiment": cfg.get("experiment_name"),
        "weights": str(weights),
        "split": "held_out_image_id_stratified",
        "n_val": len(val_df),
        **metrics,
    }
    out_path = ROOT / cfg["artifacts"]["dir"] / "eval_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("mean_dice=%.4f mean_iou=%.4f", metrics["mean_dice"], metrics["mean_iou"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
