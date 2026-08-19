"""Config + model smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from steel_defect.config import load_yaml, set_seed
from steel_defect.losses import CombinedSegLoss
from steel_defect.models import build_model

ROOT = Path(__file__).resolve().parents[1]


def test_load_configs():
    for name in ("baseline.yaml", "improved.yaml"):
        cfg = load_yaml(ROOT / "configs" / name)
        assert "train" in cfg and "model" in cfg
        assert cfg["seed"] == 42


def test_model_forward():
    set_seed(0)
    model = build_model(encoder="resnet18", encoder_weights=None, num_classes=4)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 4, 64, 64)


def test_losses():
    logits = torch.randn(2, 4, 32, 32)
    targets = torch.zeros_like(logits)
    targets[:, 0, 5:10, 5:10] = 1.0
    for mode in ("baseline", "improved"):
        loss = CombinedSegLoss(mode=mode, class_weights=[1.0, 2.0, 1.0, 1.0])
        v = loss(logits, targets)
        assert torch.isfinite(v)
        assert v.ndim == 0
