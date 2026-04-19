import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, Resized, ScaleIntensityRanged
)
import os
from monai.transforms import RandFlipd, RandRotate90d, RandZoomd, RandGaussianNoised

class OdeliaDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'uid' and potentially 'image_path'.
            transform (callable, optional): MONAI transforms to be applied.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. IDENTIFY THE IMAGE PATH
        # We check if 'image_path' was already built in predict.py
        if "image_path" in row and pd.notna(row["image_path"]):
            img_path = row["image_path"]
        else:
            # Fallback/Safety: If image_path isn't in the DF, 
            # we assume the UID is the folder name and look for Post_1
            # Note: This requires the data_root to be set correctly in your environment
            uid = str(row['uid'])
            img_path = uid # This will likely fail LoadImaged if it's just a folder

        # 2. LOAD AND TRANSFORM
        # The 'image' key matches what MONAI LoadImaged expects
        data = {"image": img_path}
        
        if self.transform:
            try:
                data = self.transform(data)
            except Exception as e:
                print(f"Error loading image at {img_path}: {e}")
                raise e

        # 3. HANDLE LABELS (For RSH, labels might be dummy/missing)
        label = int(row["label"]) if "label" in row else 0
        
        # metadata is used by predict.py to keep track of which UID produced which score
        metadata = str(row['uid'])
        
        return data["image"], label, metadata

def load_odelia_metadata(data_root):
    """
    Standard loader for the main CAM/RUMC metadata.
    """
    csv_path = os.path.join(data_root, "metadata.csv")
    df = pd.read_csv(csv_path)
    # Ensure columns are lowercase for consistency
    df.columns = [c.lower() for c in df.columns]
    return df

def get_transforms(mode="val"):
    """
    Standard preprocessing for 3D DCE-MRI.
    Added heavy augmentation for training to prevent identical predictions.
    """
    # 1. BASE TRANSFORMS (Used for both Train and Val)
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ]

    # 2. TRAINING AUGMENTATIONS (Only added if mode == "train")
    if mode == "train":
        transforms.extend([
            # Flips the image randomly across any axis
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),
            # Rotates in 90-degree increments
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            # Slight zoom in/out (90% to 110%)
            RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
            # Adds electronic noise to simulate different scanners
            RandGaussianNoised(keys=["image"], prob=0.1),
        ])

    # 3. FINAL TRANSFORMS (Used for both Train and Val)
    transforms.extend([
        Resized(keys=["image"], spatial_size=(224, 224, 80)),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True,
        ),
    ])

    return Compose(transforms)