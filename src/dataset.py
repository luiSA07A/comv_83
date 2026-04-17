import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRangePercentilesd, CropForegroundd, Resized,
    NormalizeIntensityd, RandFlipd, RandRotate90d, ToTensord
)

def load_odelia_metadata(dataset_root: str):
    # Update the root to include the /data/ subfolder
    root = Path(dataset_root) / "data" 
    all_records = []
    
    # Iterate through each center (UKA, CAM, etc.)
    for inst_dir in root.iterdir():
        if not inst_dir.is_dir(): continue
        
        anno_path = inst_dir / "metadata_unilateral" / "annotation.csv"
        split_path = inst_dir / "metadata_unilateral" / "split.csv"
        
        if anno_path.exists() and split_path.exists():
            df_anno = pd.read_csv(anno_path)
            df_split = pd.read_csv(split_path)
            df_merged = pd.merge(df_anno, df_split, on="UID")
            
            for _, row in df_merged.iterrows():
                img_path = inst_dir / "data_unilateral" / row["UID"] / "Pre.nii.gz"
                if img_path.exists():
                    all_records.append({
                        "image_path": str(img_path),
                        "label": int(row["Lesion"]),
                        "split": row["Split"],   # MUST BE "split" (lowercase) for train.py
                        "patient_id": row["UID"]
                    })
    
    final_df = pd.DataFrame(all_records)
    print(f"[Dataset] Found {len(final_df)} total records.") # Debug info
    return final_df

def get_transforms(mode="train", spatial_size=(96, 96, 32)):
    keys = ["image"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(0.7, 0.7, 3.0), mode="bilinear"), # Per challenge spec
        ScaleIntensityRangePercentilesd(keys=keys, lower=1, upper=99, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=keys, nonzero=True),
        Resized(keys=keys, spatial_size=spatial_size),
    ]
    if mode == "train":
        base += [RandFlipd(keys=keys, prob=0.5, spatial_axis=0), RandRotate90d(keys=keys, prob=0.5)]
    base.append(ToTensord(keys=keys))
    return Compose(base)

class OdeliaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = self.transform({"image": row["image_path"]})
        return data["image"], row["label"], row["patient_id"]