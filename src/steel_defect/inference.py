"""Inference helpers (PyTorch + ONNX)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from steel_defect.models import build_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(
    image_bgr: np.ndarray, image_size: int = 256
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], int, int]:
    """
    Preprocess BGR image to match dataset transforms:
    - LongestMaxSize(max_size=image_size)
    - PadIfNeeded(image_size, image_size, constant=0) with CENTER padding
    - Normalize with ImageNet mean/std

    Returns:
    - NCHW float32 batch
    - original (H, W)
    - resized (new_h, new_w) before padding
    - pad_top, pad_left (center padding offsets)
    """
    # Use the SAME albumentations pipeline as the dataset.
    # This avoids subtle mismatches in rounding/padding placement that affect crop-back.
    from steel_defect.dataset import get_transforms  # local import to avoid cycles

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_hw = (rgb.shape[0], rgb.shape[1])  # (H, W)

    tfm = get_transforms(image_size, train=False)

    # Dummy 4-channel mask: transforms will apply LongestMaxSize+PadIfNeeded consistently.
    dummy_mask = np.zeros((orig_hw[0], orig_hw[1], 4), dtype=np.float32)
    out = tfm(image=rgb, mask=dummy_mask)
    img = out["image"].astype(np.float32)  # (H=image_size, W=image_size, C=3), normalized

    # Detect padding area by comparing to the normalized value of an all-zero pixel.
    # Padded pixels before Normalize are exactly 0, so after Normalize they're constant:
    #   (0/255 - mean) / std  == (-mean/std)
    pad_const = (-IMAGENET_MEAN / IMAGENET_STD).astype(np.float32)  # (3,)
    is_pad = np.all(np.isclose(img, pad_const, atol=1e-3), axis=2)  # (H,W)
    ys, xs = np.where(~is_pad)
    if len(ys) == 0 or len(xs) == 0:
        raise RuntimeError("Could not infer non-padding region from transformed image.")

    pad_top = int(ys.min())
    pad_left = int(xs.min())
    new_h = int(ys.max() - ys.min() + 1)
    new_w = int(xs.max() - xs.min() + 1)

    # Convert to NCHW tensor for the model.
    x = np.transpose(img, (2, 0, 1))[None].astype(np.float32)  # (1,3,H,W)
    return x, orig_hw, (new_h, new_w), pad_top, pad_left


def postprocess_logits(logits: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """Backward-compatible wrapper. Prefer postprocess_logits_with_padding()."""
    raise RuntimeError("Use postprocess_logits_with_padding instead.")


def postprocess_logits_with_padding(
    logits: np.ndarray,
    orig_hw: tuple[int, int],
    resized_hw: tuple[int, int],
    pad_top: int,
    pad_left: int,
) -> np.ndarray:
    """Sigmoid + crop padding + resize each channel back to original size → (C, H, W)."""
    if logits.ndim == 4:
        logits = logits[0]
    # Stable sigmoid for large logits.
    logits = np.clip(logits, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    c = probs.shape[0]

    new_h, new_w = resized_hw
    probs_cropped = probs[
        :, pad_top : pad_top + new_h, pad_left : pad_left + new_w
    ]  # (C,new_h,new_w)

    out = np.zeros((c, orig_hw[0], orig_hw[1]), dtype=np.float32)
    for i in range(c):
        out[i] = cv2.resize(
            probs_cropped[i], (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR
        )
    return out


class TorchPredictor:
    def __init__(self, weights: Path | None, device: str = "cpu", image_size: int = 256) -> None:
        self.device = device
        self.image_size = image_size
        self.model = build_model(encoder_weights=None)
        if weights and Path(weights).exists():
            state = torch.load(weights, map_location=device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        x, orig_hw, resized_hw, pad_top, pad_left = preprocess_bgr(image_bgr, self.image_size)
        t = torch.from_numpy(x).to(self.device)
        logits = self.model(t).cpu().numpy()
        return postprocess_logits_with_padding(logits, orig_hw, resized_hw, pad_top, pad_left)


class OnnxPredictor:
    def __init__(self, onnx_path: Path, image_size: int = 256) -> None:
        import onnxruntime as ort

        self.image_size = image_size
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        x, orig_hw, resized_hw, pad_top, pad_left = preprocess_bgr(image_bgr, self.image_size)
        logits = self.session.run(None, {self.input_name: x})[0]
        return postprocess_logits_with_padding(logits, orig_hw, resized_hw, pad_top, pad_left)


def load_predictor(
    backend: str = "torch",
    weights: str | None = None,
    onnx_path: str | None = None,
    device: str = "cpu",
    image_size: int = 256,
) -> TorchPredictor | OnnxPredictor:
    if backend == "onnx":
        if not onnx_path or not Path(onnx_path).exists():
            raise FileNotFoundError("ONNX model required for backend=onnx")
        return OnnxPredictor(Path(onnx_path), image_size=image_size)
    return TorchPredictor(Path(weights) if weights else None, device=device, image_size=image_size)
