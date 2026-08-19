#!/usr/bin/env python3
"""Train UNet baseline or improved (focal + class weights) with live console progress."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
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
from steel_defect.losses import CombinedSegLoss  # noqa: E402
from steel_defect.metrics import DiceIoUAccumulator  # noqa: E402
from steel_defect.models import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device: str,
    threshold: float,
    num_classes: int = 4,
) -> dict:
    model.eval()
    acc = DiceIoUAccumulator(num_classes, threshold=threshold)
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        logits = model(images).cpu().numpy()
        # Clip before exp to avoid overflow on large logits during val.
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        acc.update(probs, masks.numpy())
    return acc.result()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    scaler: GradScaler | None,
    epoch: int,
    epochs: int,
    log_every: int,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n = 0
    t0 = time.time()
    pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=True)
    for step, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        running += float(loss.item()) * images.size(0)
        n += images.size(0)
        if step % log_every == 0 or step == len(loader):
            elapsed = time.time() - t0
            eta = elapsed / step * (len(loader) - step)
            mem = ""
            if device.startswith("cuda") and torch.cuda.is_available():
                mem = f" mem_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
            msg = f"loss={running / max(n, 1):.4f} eta_s={eta:.0f}{mem}"
            pbar.set_postfix_str(msg)
            log.info("epoch=%s step=%s/%s %s", epoch, step, len(loader), msg)
    return running / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "improved.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    device = resolve_device(str(cfg.get("device", "cuda")))
    use_amp = bool(cfg.get("amp", True)) and device == "cuda"
    log.info(
        "device=%s cuda_available=%s gpu=%s amp=%s",
        device,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        use_amp,
    )

    data_root = ROOT / cfg["data"]["root"]
    csv_path = data_root / cfg["data"]["csv"]
    images_dir = data_root / cfg["data"]["images_dir"]
    if not csv_path.exists():
        raise SystemExit(
            f"Missing {csv_path}. Run: python scripts/download_data.py --source synthetic"
        )

    index = build_image_index(csv_path, images_dir)
    holdout_path = cfg["data"].get("holdout_ids")
    holdout_ids = load_holdout_ids(ROOT / holdout_path if holdout_path else None)
    train_df, val_df, holdout_df = prepare_train_val_holdout(
        index,
        val_frac=float(cfg["data"]["val_frac"]),
        seed=int(cfg["seed"]),
        holdout_ids=holdout_ids,
    )
    log.info(
        "train_images=%s val_images=%s holdout_excluded=%s",
        len(train_df),
        len(val_df),
        len(holdout_df),
    )

    # Optional: oversample images that contain a specific class (by GT RLE presence)
    # to improve recall on rare/outlier classes.
    tcfg_pre = cfg.get("train", {})
    oversample_class_id = int(tcfg_pre.get("oversample_class_id", 0) or 0)
    oversample_factor = int(tcfg_pre.get("oversample_factor", 1) or 1)
    if oversample_class_id in (1, 2, 3, 4) and oversample_factor > 1:
        rle_col = f"rle_{oversample_class_id}"
        if rle_col not in train_df.columns:
            raise KeyError(f"Missing column {rle_col} in train_df")
        has_cls = train_df[rle_col].notna() & train_df[rle_col].astype(str).str.strip().ne("")
        to_dup = train_df[has_cls]
        if len(to_dup) > 0:
            before = len(train_df)
            train_df = pd.concat([train_df] + [to_dup] * (oversample_factor - 1), ignore_index=True)
            train_df = train_df.sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(
                drop=True
            )
            log.info(
                "oversample: class_id=%s factor=%s: train %s -> %s",
                oversample_class_id,
                oversample_factor,
                before,
                len(train_df),
            )
        else:
            log.info(
                "oversample requested but no samples contain class_id=%s in train split",
                oversample_class_id,
            )

    image_size = int(cfg["data"]["image_size"])
    train_ds = SteelDefectDataset(train_df, image_size=image_size, train=True)
    val_ds = SteelDefectDataset(val_df, image_size=image_size, train=False)
    batch_size = int(cfg["train"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 2))

    def make_loaders(bs: int):
        return (
            DataLoader(
                train_ds,
                batch_size=bs,
                shuffle=True,
                num_workers=workers,
                pin_memory=device == "cuda",
            ),
            DataLoader(
                val_ds,
                batch_size=bs,
                shuffle=False,
                num_workers=0,  # avoid Windows worker RAM spikes during cv2.imread
                pin_memory=device == "cuda",
            ),
        )

    train_loader, val_loader = make_loaders(batch_size)

    model = build_model(
        encoder=cfg["model"]["encoder"],
        encoder_weights=cfg["model"].get("encoder_weights"),
        num_classes=int(cfg["model"]["num_classes"]),
        architecture=cfg["model"].get("architecture", "unet"),
    ).to(device)

    tcfg = cfg["train"]
    criterion = CombinedSegLoss(
        mode=str(tcfg["mode"]),
        gamma=float(tcfg.get("focal_gamma", 2.0)),
        class_weights=list(tcfg.get("class_weights", [1, 1, 1, 1])),
        dice_weight=float(tcfg.get("dice_weight", 0.5)),
        bce_weight=float(tcfg.get("bce_weight", 0.5)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    scaler = GradScaler("cuda", enabled=use_amp)
    epochs = int(tcfg["epochs"])
    best_dice = -1.0
    art = cfg["artifacts"]
    weights_path = ROOT / art["weights"]
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, epochs + 1):
        try:
            tr_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                epoch,
                epochs,
                int(tcfg.get("log_every", 10)),
                use_amp,
            )
        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM — halving batch size once and retrying epoch")
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            train_loader, val_loader = make_loaders(batch_size)
            tr_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                epoch,
                epochs,
                int(tcfg.get("log_every", 10)),
                use_amp,
            )

        metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=float(cfg["eval"]["threshold"]),
            num_classes=int(cfg["model"]["num_classes"]),
        )
        log.info(
            "epoch=%s train_loss=%.4f val_mean_dice=%.4f val_mean_iou=%.4f "
            "per_class_dice=%s per_class_dice_present_only=%s n_present=%s",
            epoch,
            tr_loss,
            metrics["mean_dice"],
            metrics["mean_iou"],
            metrics["per_class_dice"],
            metrics.get("per_class_dice_present_only"),
            metrics.get("per_class_n_present"),
        )
        history.append({"epoch": epoch, "train_loss": tr_loss, **metrics})
        if metrics["mean_dice"] >= best_dice:
            best_dice = float(metrics["mean_dice"])
            torch.save(model.state_dict(), weights_path)
            log.info("saved best checkpoint dice=%.4f → %s", best_dice, weights_path)

    out_metrics = {
        "experiment": cfg.get("experiment_name"),
        "mode": tcfg["mode"],
        "seed": cfg["seed"],
        "device": device,
        "best_mean_dice": best_dice,
        "history": history,
        "imbalance_strategy": (
            "uniform Dice+BCE"
            if tcfg["mode"] == "baseline"
            else (
                f"focal(gamma={tcfg.get('focal_gamma')}) + "
                f"class_weights={tcfg.get('class_weights')}"
            )
        ),
        "weights": str(weights_path.relative_to(ROOT)),
    }
    metrics_path = ROOT / art["metrics"]
    metrics_path.write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")
    log.info("Wrote %s", metrics_path)
    print(json.dumps({"best_mean_dice": best_dice, "weights": str(weights_path)}, indent=2))


if __name__ == "__main__":
    main()
