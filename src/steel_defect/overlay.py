"""Overlay predicted masks on RGB images for API/demo."""

from __future__ import annotations

import numpy as np

# Distinct colors for classes 1..4 (RGB)
CLASS_COLORS = np.array(
    [
        [255, 64, 64],
        [64, 255, 64],
        [64, 128, 255],
        [255, 220, 64],
    ],
    dtype=np.uint8,
)


def overlay_masks(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend multi-channel masks onto RGB uint8 image.

    masks: (C, H, W) probabilities or logits-already-sigmoided.
    """
    if image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb must be uint8")
    h, w = image_rgb.shape[:2]
    if masks.shape[-2:] != (h, w):
        raise ValueError(f"mask spatial {masks.shape[-2:]} != image {(h, w)}")

    out = image_rgb.astype(np.float32).copy()
    for ci in range(masks.shape[0]):
        bin_m = masks[ci] >= threshold
        if not bin_m.any():
            continue
        color = CLASS_COLORS[ci % len(CLASS_COLORS)].astype(np.float32)
        out[bin_m] = (1 - alpha) * out[bin_m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)
