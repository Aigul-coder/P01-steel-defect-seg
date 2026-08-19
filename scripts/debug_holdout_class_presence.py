#!/usr/bin/env python3
"""
Debug: how many holdout images have non-empty GT for each class after dataset transforms.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steel_defect.config import load_yaml  # noqa: E402
from steel_defect.dataset import build_image_index, load_holdout_ids, SteelDefectDataset  # noqa: E402


def main() -> None:
    cfg = load_yaml(ROOT / "configs" / "baseline.yaml")
    data_root = ROOT / cfg["data"]["root"]
    index = build_image_index(data_root / cfg["data"]["csv"], data_root / cfg["data"]["images_dir"])
    holdout_ids = load_holdout_ids(ROOT / cfg["data"]["holdout_ids"])
    holdout_df = index[index["ImageId"].astype(str).isin(holdout_ids)].reset_index(drop=True)

    ds = SteelDefectDataset(holdout_df, image_size=int(cfg["data"]["image_size"]), train=False)

    present: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for i in range(len(ds)):
        s = ds[i]
        m = s["mask"].numpy()  # (C,H,W)
        for ci in range(4):
            pos = float((m[ci] >= 0.5).mean())
            if pos > 0:
                present[ci + 1].append(s["image_id"])

    for cid in range(1, 5):
        print(f"class {cid}: n_present={len(present[cid])}/{len(ds)}")
        if present[cid]:
            print("  ids:", ", ".join(present[cid]))


if __name__ == "__main__":
    main()

