"""Dataset and augmentations for Severstal CSV + images."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from steel_defect.rle import masks_to_multichannel


def build_image_index(csv_path: Path, images_dir: Path) -> pd.DataFrame:
    """Collapse train.csv rows into one row per ImageId with class RLEs."""
    df = pd.read_csv(csv_path)
    # Expected columns: ImageId, ClassId, EncodedPixels
    if "EncodedPixels" not in df.columns:
        raise ValueError("train.csv must contain EncodedPixels")
    records: list[dict] = []
    for image_id, group in df.groupby("ImageId"):
        path = images_dir / str(image_id)
        if not path.exists():
            continue
        rles: dict[int, str | None] = {1: None, 2: None, 3: None, 4: None}
        has_defect = 0
        for _, row in group.iterrows():
            cid = int(row["ClassId"])
            ep = row.get("EncodedPixels")
            if pd.isna(ep) or str(ep).strip() == "":
                continue
            rles[cid] = str(ep)
            has_defect = 1
        records.append(
            {
                "ImageId": image_id,
                "path": str(path),
                "has_defect": has_defect,
                "rle_1": rles[1],
                "rle_2": rles[2],
                "rle_3": rles[3],
                "rle_4": rles[4],
            }
        )
    return pd.DataFrame.from_records(records)


def stratified_subset(index: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample ~n images stratified by has_defect."""
    if len(index) <= n:
        return index.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pos = index[index["has_defect"] == 1]
    neg = index[index["has_defect"] == 0]
    n_pos = min(len(pos), max(1, n // 2))
    n_neg = min(len(neg), n - n_pos)
    if n_pos + n_neg < n and len(pos) > n_pos:
        n_pos = min(len(pos), n - n_neg)
    pos_idx = rng.choice(pos.index.to_numpy(), size=n_pos, replace=False) if n_pos else []
    neg_idx = rng.choice(neg.index.to_numpy(), size=n_neg, replace=False) if n_neg else []
    chosen = index.loc[list(pos_idx) + list(neg_idx)]
    return chosen.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_val_split(
    index: pd.DataFrame, val_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Image-level split stratified by defect presence (no plate grouping available)."""
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _, group in index.groupby("has_defect"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        if len(train_idx) == 0:
            train_idx, val_idx = idx[:1], idx[1:]
        train_parts.append(index.loc[train_idx])
        if len(val_idx):
            val_parts.append(index.loc[val_idx])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = (
        pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if val_parts
        else train_df.iloc[:0].copy()
    )
    return train_df, val_df


def load_holdout_ids(path: Path | None) -> set[str]:
    """Load ImageIds to exclude from train and val (one id per line, # comments allowed)."""
    if path is None or not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(line)
    return ids


def exclude_holdout(
    index: pd.DataFrame, holdout_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove holdout images from the index used for train/val splitting."""
    if not holdout_ids:
        return index.reset_index(drop=True), index.iloc[:0].copy()
    mask = index["ImageId"].astype(str).isin(holdout_ids)
    holdout_df = index.loc[mask].reset_index(drop=True)
    remaining = index.loc[~mask].reset_index(drop=True)
    missing = holdout_ids - set(holdout_df["ImageId"].astype(str))
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"holdout ids not found in dataset index: {preview}")
    return remaining, holdout_df


def prepare_train_val_holdout(
    index: pd.DataFrame,
    val_frac: float,
    seed: int,
    holdout_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split index into train/val after carving out a blind holdout set."""
    remaining, holdout_df = exclude_holdout(index, holdout_ids or set())
    train_df, val_df = train_val_split(remaining, val_frac, seed)
    return train_df, val_df, holdout_df


def get_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.15),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SteelDefectDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int = 256, train: bool = True) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tfm = get_transforms(image_size, train=train)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        image = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        rles = {
            1: row["rle_1"],
            2: row["rle_2"],
            3: row["rle_3"],
            4: row["rle_4"],
        }
        mask = masks_to_multichannel(rles, h, w)  # C,H,W
        mask_hwc = np.transpose(mask, (1, 2, 0))
        out = self.tfm(image=image, mask=mask_hwc)
        img = out["image"].astype(np.float32)
        msk = out["mask"].astype(np.float32)
        if msk.ndim == 2:
            msk = msk[..., None]
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        msk_t = torch.from_numpy(msk).permute(2, 0, 1).contiguous()
        return {"image": img_t, "mask": msk_t, "image_id": str(row["ImageId"])}
