"""Model builders: UNet via segmentation-models-pytorch."""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    num_classes: int = 4,
    architecture: str = "unet",
) -> nn.Module:
    kwargs: dict[str, Any] = {
        "encoder_name": encoder,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": num_classes,
        "activation": None,
    }
    if architecture.lower() == "fpn":
        return smp.FPN(**kwargs)
    return smp.Unet(**kwargs)
