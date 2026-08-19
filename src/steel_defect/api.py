"""FastAPI serving: health + predict with optional mask overlay."""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from steel_defect.inference import load_predictor
from steel_defect.overlay import overlay_masks

THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
BACKEND = os.getenv("INFERENCE_BACKEND", "torch")
WEIGHTS = os.getenv("WEIGHTS_PATH", "artifacts/model.pt")
ONNX_PATH = os.getenv("ONNX_PATH", "artifacts/model.onnx")
DEVICE = os.getenv("DEVICE", "cpu")

_predictor: Any = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str


class PredictResponse(BaseModel):
    classes: list[dict[str, float | int]]
    mean_positive_area: float
    overlay_png_base64: str | None = Field(default=None)


def _ensure_predictor() -> Any:
    global _predictor
    if _predictor is None:
        try:
            _predictor = load_predictor(
                backend=BACKEND,
                weights=WEIGHTS if PathExists(WEIGHTS) else None,
                onnx_path=ONNX_PATH,
                device=DEVICE,
                image_size=IMAGE_SIZE,
            )
        except Exception:
            # Random-init torch model for demos without weights
            _predictor = load_predictor(
                backend="torch", weights=None, device=DEVICE, image_size=IMAGE_SIZE
            )
    return _predictor


def PathExists(p: str) -> bool:
    from pathlib import Path

    return Path(p).exists()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _ensure_predictor()
    yield


app = FastAPI(title="P01 Steel Defect Segmentation", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_predictor is not None, backend=BACKEND)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    return_overlay: bool = True,
) -> PredictResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    predictor = _ensure_predictor()
    probs = predictor.predict(bgr)  # C,H,W
    classes = []
    areas = []
    for i in range(probs.shape[0]):
        area = float((probs[i] >= THRESHOLD).mean())
        areas.append(area)
        classes.append(
            {"class_id": i + 1, "positive_area_frac": area, "max_prob": float(probs[i].max())}
        )

    overlay_b64 = None
    if return_overlay:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        over = overlay_masks(rgb, probs, threshold=THRESHOLD)
        buf = io.BytesIO()
        Image.fromarray(over).save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return PredictResponse(
        classes=classes,
        mean_positive_area=float(np.mean(areas)),
        overlay_png_base64=overlay_b64,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "steel_defect.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
