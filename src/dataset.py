"""
TDT4265 Mini-Project - Option 2: Breast Cancer MRI Classification
Dataset loading and preprocessing using MONAI
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    ToTensord,
    NormalizeIntensityd,
)


# ─── Label mapping ────────────────────────────────────────────────────────────
LABEL_MAP = {"normal": 0, "benign": 1, "malignant": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def load_odelia_metadata(dataset_root: str) -> pd.DataFrame:
    """
    Parse the ODELIA2025 dataset folder structure into a flat DataFrame.
    Expected structure:
        dataset_root/
            <subset>/          e.g. UKA, UKE, RSH ...
                <patient_id>/
                    left/   -> DICOM or NIfTI volumes
                    right/  -> DICOM or NIfTI volumes
                labels.csv  OR  labels.json
    Returns columns: [patient_id, subset, side, label_int, image_path]
    """
    records = []
    root = Path(dataset_root)

    for subset_dir in sorted(root.iterdir()):
        if not subset_dir.is_dir():
            continue
        subset = subset_dir.name

        # Try to find a label file (adjust to actual format in the dataset)
        label_file_csv = subset_dir / "labels.csv"
        label_file_json = subset_dir / "labels.json"

        labels_dict = {}  # patient_id -> {left: str, right: str}

        if label_file_csv.exists():
            df_lbl = pd.read_csv(label_file_csv)
            for _, row in df_lbl.iterrows():
                labels_dict[str(row["patient_id"])] = {
                    "left": row.get("left", "normal"),
                    "right": row.get("right", "normal"),
                }
        elif label_file_json.exists():
            with open(label_file_json) as f:
                labels_dict = json.load(f)

        for patient_dir in sorted(subset_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name

            for side in ["left", "right"]:
                side_dir = patient_dir / side
                if not side_dir.exists():
                    continue

                # Find first NIfTI file in folder (adjust if DICOM)
                nii_files = list(side_dir.glob("*.nii.gz")) + list(side_dir.glob("*.nii"))
                if not nii_files:
                    continue
                image_path = str(nii_files[0])

                label_str = labels_dict.get(pid, {}).get(side, None)
                label_int = LABEL_MAP.get(label_str, -1) if label_str else -1

                records.append({
                    "patient_id": pid,
                    "subset": subset,
                    "side": side,
                    "label_str": label_str,
                    "label_int": label_int,
                    "image_path": image_path,
                })

    df = pd.DataFrame(records)
    print(f"[Dataset] Loaded {len(df)} breast entries from {df['subset'].nunique()} subsets.")
    print(df["label_str"].value_counts())
    return df


def get_transforms(mode: str = "train", spatial_size=(96, 96, 32)):
    """
    Returns MONAI transform pipeline.
    mode: 'train' | 'val' | 'test'
    """
    keys = ["image"]
    shared = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys=keys, lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=keys, source_key="image"),
        Resized(keys=keys, spatial_size=spatial_size),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
    ]

    augment = [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.3, max_k=3),
        RandZoomd(keys=keys, prob=0.3, min_zoom=0.9, max_zoom=1.1),
        RandGaussianNoised(keys=keys, prob=0.2, std=0.01),
        RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.8, 1.2)),
    ]

    final = [ToTensord(keys=keys)]

    if mode == "train":
        return Compose(shared + augment + final)
    else:
        return Compose(shared + final)


class OdeliaDataset(Dataset):
    """
    PyTorch Dataset for ODELIA breast MRI classification.
    Each sample is one breast (left or right) with a label.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        # Only keep rows with valid labels
        self.df = df[df["label_int"] >= 0].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {"image": row["image_path"]}

        if self.transform:
            data = self.transform(data)

        label = torch.tensor(int(row["label_int"]), dtype=torch.long)
        return data["image"], label, row["patient_id"], row["side"]
