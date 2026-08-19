"""Losses for imbalanced multi-label segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalBCEWithLogits(nn.Module):
    """Per-pixel focal BCE for multi-label masks."""

    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, pos_weight: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(4))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: (N, C, H, W)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        focal = (self.alpha * (1 - pt) ** self.gamma) * bce
        # channel weights broadcast
        w = self.pos_weight.view(1, -1, 1, 1)
        return (focal * w).mean()


class CombinedSegLoss(nn.Module):
    """Dice + focal BCE (improved) or Dice + plain BCE (baseline)."""

    def __init__(
        self,
        mode: str = "improved",
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        weights = torch.tensor(class_weights or [1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        if mode == "baseline":
            self.bce: nn.Module = nn.BCEWithLogitsLoss(pos_weight=weights)
            self._focal = False
        else:
            self.bce = FocalBCEWithLogits(gamma=gamma, pos_weight=weights)
            self._focal = True

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._focal:
            bce = self.bce(logits, targets)
        else:
            # BCEWithLogitsLoss expects pos_weight (C,) and logits (N,C,H,W) — OK in recent torch
            n, c, h, w = logits.shape
            bce = self.bce(
                logits.permute(0, 2, 3, 1).reshape(-1, c),
                targets.permute(0, 2, 3, 1).reshape(-1, c),
            )
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * bce
