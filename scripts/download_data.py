#!/usr/bin/env python3
"""Download Severstal data (Kaggle / HF) or generate synthetic smoke set."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import shutil
import sys
import time
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect.dataset import build_image_index, stratified_subset  # noqa: E402
from steel_defect.rle import rle_encode  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_data")


def _write_synthetic(
    out_root: Path, n: int, seed: int, height: int = 256, width: int = 1600
) -> None:
    rng = np.random.default_rng(seed)
    img_dir = out_root / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in tqdm(range(n), desc="synthetic images"):
        image_id = f"synth_{i:04d}.jpg"
        img = np.full((height, width, 3), 180, dtype=np.uint8)
        img = img + rng.integers(-20, 20, size=img.shape, dtype=np.int16)
        img = np.clip(img, 0, 255).astype(np.uint8)
        has_defect = i % 2 == 0
        class_ids: list[int] = []
        if has_defect:
            class_ids = [int(rng.integers(1, 5))]
            if rng.random() < 0.3:
                class_ids.append(int(rng.integers(1, 5)))
            class_ids = sorted(set(class_ids))
        for cid in class_ids:
            mask = np.zeros((height, width), dtype=np.uint8)
            y0 = int(rng.integers(10, height - 40))
            x0 = int(rng.integers(10, width - 80))
            y1 = y0 + int(rng.integers(8, 30))
            x1 = x0 + int(rng.integers(40, 120))
            mask[y0:y1, x0:x1] = 1
            img[mask > 0] = (40, 40, 40)
            rows.append({"ImageId": image_id, "ClassId": cid, "EncodedPixels": rle_encode(mask)})
        if not class_ids:
            rows.append({"ImageId": image_id, "ClassId": 1, "EncodedPixels": ""})
        cv2.imwrite(str(img_dir / image_id), img)
    pd.DataFrame(rows).to_csv(out_root / "train.csv", index=False)
    log.info("Wrote synthetic set n=%s → %s", n, out_root)


def _download_kaggle(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit(
            "kaggle package missing. pip install kaggle and place ~/.kaggle/kaggle.json"
        ) from e
    api = KaggleApi()
    api.authenticate()
    log.info("Downloading severstal-steel-defect-detection (train only)…")
    # Competition files — large; progress via kaggle CLI verbosity
    api.competition_download_file(
        "severstal-steel-defect-detection",
        "train.csv",
        path=str(raw_dir),
        quiet=False,
    )
    api.competition_download_file(
        "severstal-steel-defect-detection",
        "train_images.zip",
        path=str(raw_dir),
        quiet=False,
    )
    zip_path = raw_dir / "train_images.zip"
    if zip_path.exists():
        log.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir / "train_images")
        # flatten if nested
        nested = raw_dir / "train_images" / "train_images"
        if nested.is_dir():
            for p in nested.iterdir():
                shutil.move(str(p), str(raw_dir / "train_images" / p.name))
            nested.rmdir()


def _decode_fo_mask(mask_doc: dict) -> np.ndarray:
    """FiftyOne hub stores masks as zlib-compressed .npy (BSON $binary)."""
    b64 = mask_doc["$binary"]["base64"]
    raw = base64.b64decode(b64)
    return np.load(io.BytesIO(zlib.decompress(raw)))


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize_from_fiftyone(snapshot: Path, raw_dir: Path) -> None:
    """Flatten HF FiftyOne dump into train.csv + train_images/ (+ test_images/)."""
    samples_path = snapshot / "samples.json"
    with open(samples_path, encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload["samples"] if isinstance(payload, dict) else payload

    train_dir = raw_dir / "train_images"
    test_dir = raw_dir / "test_images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_train = n_test = 0
    for sample in tqdm(samples, desc="materialize images+RLE"):
        rel = sample["filepath"].replace("\\", "/")
        src = snapshot / rel
        image_id = sample.get("image_id") or Path(rel).name
        split = sample.get("split", "train")
        dest_dir = train_dir if split == "train" else test_dir
        if not src.exists():
            log.warning("missing image %s", src)
            continue
        _link_or_copy(src, dest_dir / image_id)
        if split != "train":
            n_test += 1
            continue
        n_train += 1
        gt = sample.get("ground_truth")
        if not gt or not sample.get("has_defect"):
            rows.append({"ImageId": image_id, "ClassId": 1, "EncodedPixels": ""})
            continue
        mask = _decode_fo_mask(gt["mask"])
        classes = sample.get("defect_classes") or []
        wrote = False
        for cid in sorted(set(int(c) for c in classes)):
            binary = (mask == cid).astype(np.uint8)
            if binary.sum() == 0:
                continue
            rows.append(
                {
                    "ImageId": image_id,
                    "ClassId": cid,
                    "EncodedPixels": rle_encode(binary),
                }
            )
            wrote = True
        if not wrote:
            rows.append({"ImageId": image_id, "ClassId": 1, "EncodedPixels": ""})

    pd.DataFrame(rows).to_csv(raw_dir / "train.csv", index=False)
    log.info("Materialized train=%s test=%s csv_rows=%s → %s", n_train, n_test, len(rows), raw_dir)


def _ensure_samples_json(snapshot: Path) -> Path:
    dest = snapshot / "samples.json"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        "Voxel51/severstal_steel_defects",
        "samples.json",
        repo_type="dataset",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(cached).resolve() != dest.resolve():
        shutil.copy2(cached, dest)
    return dest


def _hf_download_file(filename: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 20.0
    for attempt in range(1, 12):
        try:
            path = hf_hub_download(
                repo_id="Voxel51/severstal_steel_defects",
                filename=filename,
                repo_type="dataset",
            )
            src = Path(path)
            if src.resolve() != dest.resolve():
                _link_or_copy(src, dest)
            return
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                log.warning("429 on %s attempt %s, sleep %.0fs", filename, attempt, delay)
                time.sleep(delay)
                delay = min(delay * 1.5, 180.0)
                continue
            raise
    raise RuntimeError(f"failed to download {filename} after retries")


def _download_hf(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot = raw_dir / "_hf"
    snapshot.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    log.info("Downloading HF dataset Voxel51/severstal_steel_defects → %s", snapshot)

    samples_path = _ensure_samples_json(snapshot)
    with open(samples_path, encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload["samples"] if isinstance(payload, dict) else payload
    rels = [str(s["filepath"]).replace("\\", "/") for s in samples]
    missing = [rel for rel in rels if not (snapshot / rel).exists()]
    log.info("images total=%s missing=%s", len(rels), len(missing))
    for rel in tqdm(missing, desc="hf images"):
        _hf_download_file(rel, snapshot / rel)

    still_missing = sum(1 for rel in rels if not (snapshot / rel).exists())
    if still_missing:
        raise SystemExit(f"HF download incomplete: {still_missing} images missing")

    marker = raw_dir / "HF_SNAPSHOT_PATH.txt"
    marker.write_text(str(snapshot.resolve()), encoding="utf-8")
    _materialize_from_fiftyone(snapshot, raw_dir)


def _make_subset(raw_root: Path, subset_root: Path, n: int, seed: int) -> None:
    csv_path = raw_root / "train.csv"
    images_dir = raw_root / "train_images"
    if not csv_path.exists() or not images_dir.exists():
        raise FileNotFoundError(f"Need {csv_path} and {images_dir}")
    index = build_image_index(csv_path, images_dir)
    sub = stratified_subset(index, n=n, seed=seed)
    subset_root.mkdir(parents=True, exist_ok=True)
    out_img = subset_root / "train_images"
    out_img.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="copy subset"):
        src = Path(row["path"])
        dst = out_img / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        for cid, key in enumerate(["rle_1", "rle_2", "rle_3", "rle_4"], start=1):
            ep = row[key]
            if ep is None or (isinstance(ep, float) and np.isnan(ep)) or str(ep).strip() == "":
                continue
            rows.append({"ImageId": row["ImageId"], "ClassId": cid, "EncodedPixels": ep})
        if row["has_defect"] == 0:
            rows.append({"ImageId": row["ImageId"], "ClassId": 1, "EncodedPixels": ""})
    pd.DataFrame(rows).to_csv(subset_root / "train.csv", index=False)
    log.info("Subset size=%s → %s", len(sub), subset_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("kaggle", "hf", "synthetic"), default="synthetic")
    ap.add_argument(
        "--subset",
        type=int,
        default=0,
        help="Stratified smoke subset size; 0=skip (full set only)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--subset-dir", type=Path, default=ROOT / "data" / "subset")
    args = ap.parse_args()

    if args.source == "synthetic":
        n = args.subset if args.subset > 0 else 32
        _write_synthetic(args.subset_dir, n=n, seed=args.seed)
        return

    if args.source == "kaggle":
        _download_kaggle(args.raw_dir)
    else:
        _download_hf(args.raw_dir)

    if args.subset and args.subset > 0:
        _make_subset(args.raw_dir, args.subset_dir, n=args.subset, seed=args.seed)
    log.info("Done. See data/README.md for license notes.")


if __name__ == "__main__":
    main()
