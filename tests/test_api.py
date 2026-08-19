"""API tests with TestClient (random-init weights OK)."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from steel_defect.api import app


def _png_bytes(w: int = 64, h: int = 48) -> bytes:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = (120, 130, 140)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "backend" in body


def test_predict():
    with TestClient(app) as client:
        files = {"file": ("t.png", _png_bytes(), "image/png")}
        r = client.post("/predict", files=files)
        assert r.status_code == 200
        body = r.json()
        assert len(body["classes"]) == 4
        assert "overlay_png_base64" in body
        assert body["overlay_png_base64"]
