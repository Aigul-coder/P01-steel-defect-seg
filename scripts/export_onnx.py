#!/usr/bin/env python3
"""Export PyTorch segmentation model to ONNX and bench CPU latency."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect.config import load_yaml, resolve_device  # noqa: E402
from steel_defect.models import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_onnx")


def bench_onnx(onnx_path: Path, image_size: int, runs: int = 50) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    for _ in range(5):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "improved.yaml")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    device = resolve_device("cpu")  # export on CPU for portability
    image_size = int(cfg["data"]["image_size"])

    weights = Path(args.weights) if args.weights else ROOT / cfg["artifacts"]["weights"]
    out = Path(args.out) if args.out else ROOT / cfg["export"]["onnx_path"]
    out.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(
        encoder=cfg["model"]["encoder"],
        encoder_weights=None,
        num_classes=int(cfg["model"]["num_classes"]),
        architecture=cfg["model"].get("architecture", "unet"),
    )
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        log.info("Loaded %s", weights)
    else:
        log.warning("Weights missing (%s) — exporting randomly initialized model", weights)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": int(cfg["export"].get("opset", 17)),
    }
    # Prefer legacy exporter when available (torch>=2.6 may default to dynamo).
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **export_kwargs)
    log.info("Wrote %s", out)
    latency = bench_onnx(out, image_size=image_size, runs=args.runs)
    log.info("ONNX CPU latency ms: %s", latency)
    meta = {
        "onnx": str(out.relative_to(ROOT)),
        "onnx_cpu_latency_ms": latency,
        "image_size": image_size,
    }
    meta_path = ROOT / cfg["artifacts"]["dir"] / "onnx_latency.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
