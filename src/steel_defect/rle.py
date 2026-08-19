"""RLE encode/decode for Severstal-style run-length masks (column-major)."""

from __future__ import annotations

import numpy as np


def rle_decode(rle: str | float | None, height: int, width: int) -> np.ndarray:
    """Decode Severstal RLE string to binary mask (H, W), uint8 {0,1}.

    Empty / NaN RLE → all zeros. Encoding is column-major (Fortran order).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return mask.reshape((height, width), order="F")
    s = str(rle).strip()
    if not s or s.lower() == "nan":
        return mask.reshape((height, width), order="F")

    values = list(map(int, s.split()))
    starts, lengths = values[0::2], values[1::2]
    for start, length in zip(starts, lengths, strict=True):
        start_idx = start - 1  # 1-based
        end_idx = start_idx + length
        mask[start_idx:end_idx] = 1
    return mask.reshape((height, width), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode binary mask (H, W) to Severstal RLE (column-major, 1-based)."""
    pixels = mask.reshape(mask.shape[0] * mask.shape[1], order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks_to_multichannel(
    rles_by_class: dict[int, str | None],
    height: int,
    width: int,
    num_classes: int = 4,
) -> np.ndarray:
    """Build (C, H, W) float32 mask from class→RLE map (classes 1..num_classes)."""
    out = np.zeros((num_classes, height, width), dtype=np.float32)
    for cls in range(1, num_classes + 1):
        rle = rles_by_class.get(cls)
        out[cls - 1] = rle_decode(rle, height, width).astype(np.float32)
    return out
