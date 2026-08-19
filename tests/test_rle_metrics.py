"""Unit tests for RLE and metrics (no heavy downloads)."""

from __future__ import annotations

import numpy as np

from steel_defect.dataset import prepare_train_val_holdout
from steel_defect.metrics import DiceIoUAccumulator, dice_iou_per_class
from steel_defect.overlay import overlay_masks
from steel_defect.rle import masks_to_multichannel, rle_decode, rle_encode


def test_rle_roundtrip():
    h, w = 64, 128
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 30:50] = 1
    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, h, w)
    assert np.array_equal(mask, decoded)


def test_rle_empty():
    m = rle_decode("", 16, 16)
    assert m.shape == (16, 16)
    assert m.sum() == 0
    assert rle_decode(None, 8, 8).sum() == 0


def test_masks_to_multichannel():
    h, w = 32, 32
    rle = rle_encode(np.ones((h, w), dtype=np.uint8))
    multi = masks_to_multichannel({1: rle, 2: None, 3: None, 4: None}, h, w)
    assert multi.shape == (4, h, w)
    assert multi[0].sum() == h * w
    assert multi[1].sum() == 0


def test_holdout_excluded_from_train_and_val():
    import pandas as pd

    index = pd.DataFrame(
        {
            "ImageId": [f"{i:03d}.jpg" for i in range(100)],
            "path": [f"/tmp/{i:03d}.jpg" for i in range(100)],
            "has_defect": [i % 2 for i in range(100)],
            "rle_1": [None] * 100,
            "rle_2": [None] * 100,
            "rle_3": [None] * 100,
            "rle_4": [None] * 100,
        }
    )
    holdout = {f"{i:03d}.jpg" for i in range(10)}
    train_df, val_df, holdout_df = prepare_train_val_holdout(index, 0.2, 42, holdout)
    assert len(holdout_df) == 10
    train_ids = set(train_df["ImageId"])
    val_ids = set(val_df["ImageId"])
    assert holdout.isdisjoint(train_ids)
    assert holdout.isdisjoint(val_ids)


def test_dice_accumulator_matches_batch():
    rng = np.random.default_rng(0)
    preds = rng.random((12, 4, 32, 32), dtype=np.float32)
    tgts = (rng.random((12, 4, 32, 32)) > 0.7).astype(np.float32)
    full = dice_iou_per_class(preds, tgts)
    acc = DiceIoUAccumulator(4)
    acc.update(preds[:5], tgts[:5])
    acc.update(preds[5:], tgts[5:])
    streamed = acc.result()
    assert abs(full["mean_dice"] - streamed["mean_dice"]) < 1e-6
    assert abs(full["mean_iou"] - streamed["mean_iou"]) < 1e-6


def test_dice_perfect():
    tgt = np.zeros((1, 4, 16, 16), dtype=np.float32)
    tgt[0, 0, 2:8, 2:8] = 1
    pred = tgt.copy()
    m = dice_iou_per_class(pred, tgt)
    assert m["per_class_dice"]["1"] > 0.99
    assert m["mean_dice"] > 0.2


def test_overlay_shape():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    masks = np.zeros((4, 40, 50), dtype=np.float32)
    masks[0, 5:15, 5:15] = 0.9
    out = overlay_masks(img, masks)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
