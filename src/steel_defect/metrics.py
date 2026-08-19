"""Segmentation metrics: Dice and IoU per class + means."""

from __future__ import annotations

import numpy as np


def _binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    inter = float(np.logical_and(pred, target).sum())
    denom = float(pred.sum() + target.sum())
    return float((2.0 * inter + eps) / (denom + eps))


def _binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    inter = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return float((inter + eps) / (union + eps))


class DiceIoUAccumulator:
    """Streaming mean per-image Dice/IoU without holding the full val set in RAM."""

    def __init__(self, num_classes: int, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        self.eps = eps
        self.dice_sum = np.zeros(num_classes, dtype=np.float64)
        self.iou_sum = np.zeros(num_classes, dtype=np.float64)
        self.dice_present_sum = np.zeros(num_classes, dtype=np.float64)
        self.iou_present_sum = np.zeros(num_classes, dtype=np.float64)
        self.n_present = np.zeros(num_classes, dtype=np.int64)
        self.n_images = 0

    def update(self, preds: np.ndarray, targets: np.ndarray) -> None:
        if preds.ndim == 3:
            preds = preds[None]
            targets = targets[None]
        if preds.shape != targets.shape:
            raise ValueError(f"shape mismatch: {preds.shape} vs {targets.shape}")

        n, c, _, _ = preds.shape
        if c != self.num_classes:
            raise ValueError(f"expected {self.num_classes} classes, got {c}")

        pred_bin = (preds >= self.threshold).astype(np.uint8)
        tgt_bin = (targets >= 0.5).astype(np.uint8)
        for i in range(n):
            for ci in range(c):
                dice = _binary_dice(pred_bin[i, ci], tgt_bin[i, ci], self.eps)
                iou = _binary_iou(pred_bin[i, ci], tgt_bin[i, ci], self.eps)
                self.dice_sum[ci] += dice
                self.iou_sum[ci] += iou
                if tgt_bin[i, ci].any():
                    self.dice_present_sum[ci] += dice
                    self.iou_present_sum[ci] += iou
                    self.n_present[ci] += 1
            self.n_images += 1

    def result(self) -> dict[str, dict[str, float] | float | dict[str, int]]:
        if self.n_images == 0:
            return {
                "mean_dice": 0.0,
                "mean_iou": 0.0,
                "per_class_dice": {},
                "per_class_iou": {},
                "per_class_dice_present_only": {},
                "per_class_iou_present_only": {},
                "per_class_n_present": {},
            }

        dice_scores = {
            str(ci + 1): float(self.dice_sum[ci] / self.n_images) for ci in range(self.num_classes)
        }
        iou_scores = {
            str(ci + 1): float(self.iou_sum[ci] / self.n_images) for ci in range(self.num_classes)
        }
        dice_present = {
            str(ci + 1): float(self.dice_present_sum[ci] / self.n_present[ci])
            if self.n_present[ci] > 0
            else 0.0
            for ci in range(self.num_classes)
        }
        iou_present = {
            str(ci + 1): float(self.iou_present_sum[ci] / self.n_present[ci])
            if self.n_present[ci] > 0
            else 0.0
            for ci in range(self.num_classes)
        }
        present_counts = {str(ci + 1): int(self.n_present[ci]) for ci in range(self.num_classes)}
        present_classes = [ci for ci in range(self.num_classes) if self.n_present[ci] > 0]
        mean_dice_present = (
            float(np.mean([dice_present[str(ci + 1)] for ci in present_classes]))
            if present_classes
            else 0.0
        )
        mean_iou_present = (
            float(np.mean([iou_present[str(ci + 1)] for ci in present_classes]))
            if present_classes
            else 0.0
        )
        return {
            "per_class_dice": dice_scores,
            "per_class_iou": iou_scores,
            "mean_dice": float(np.mean(list(dice_scores.values()))),
            "mean_iou": float(np.mean(list(iou_scores.values()))),
            "per_class_dice_present_only": dice_present,
            "per_class_iou_present_only": iou_present,
            "per_class_n_present": present_counts,
            "mean_dice_present_only": mean_dice_present,
            "mean_iou_present_only": mean_iou_present,
        }


def dice_iou_per_class(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, float] | float]:
    """Compute per-class Dice/IoU for multi-label masks.

    preds/targets: (N, C, H, W) or (C, H, W) float probabilities / binary.
    """
    num_classes = preds.shape[0] if preds.ndim == 3 else preds.shape[1]
    acc = DiceIoUAccumulator(num_classes, threshold=threshold)
    acc.update(preds, targets)
    return acc.result()
