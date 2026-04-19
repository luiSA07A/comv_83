import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, Resized, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandZoomd, RandGaussianNoised
)
import os

class OdeliaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        
        data = {"image": img_path}
        
        if self.transform:
            try:
                data = self.transform(data)
            except Exception as e:
                # If an image is missing, this helps you find which one
                print(f"Error loading image at {img_path}: {e}")
                raise e

        # Using the column we created in load_odelia_metadata
        label = int(row["label"])
        uid = str(row['uid'])
        
        return data["image"], label, uid

def load_odelia_metadata(data_root):
    """
    Robust loader that merges institutional annotations with the main split file.
    """
    # 1. Load the main split file (this contains UID, Split, Institution)
    split_file = os.path.join(data_root, "split_unilateral.csv")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Could not find {split_file}")
    
    df_split = pd.read_csv(split_file)
    
    # 2. Gather annotations (labels) from all institutions
    all_annotations = []
    institutions = df_split['Institution'].unique()
    
    for inst in institutions:
        # Path: data_root/data/INST/metadata_unilateral/annotation.csv
        anno_path = os.path.join(data_root, "data", inst, "metadata_unilateral", "annotation.csv")
        if os.path.exists(anno_path):
            df_anno = pd.read_csv(anno_path)
            all_annotations.append(df_anno)
            
    if not all_annotations:
        raise ValueError("Found no annotation files in institutional folders!")
        
    df_all_anno = pd.concat(all_annotations).drop_duplicates(subset=['UID'])
    
    # 3. Merge splits with labels (Lesion -> label)
    df = pd.merge(df_split, df_all_anno[['UID', 'Lesion']], on='UID', how='inner')
    
    # 4. Standardize for train.py (lowercase and renamed)
    df = df.rename(columns={'UID': 'uid', 'Split': 'split', 'Lesion': 'label'})
    df['split'] = df['split'].str.lower()
    
    # 5. Construct full image paths
    # Structure: data_root/data/Institution/UID/Post_1.nii.gz
    def get_path(row):
        return os.path.join(data_root, "data", row['Institution'], row['uid'], "Post_1.nii.gz")
    
    df['image_path'] = df.apply(get_path, axis=1)
    
    # Optional: Verify images exist to avoid crashes mid-training
    # df = df[df['image_path'].map(os.path.exists)]
    
    print(f"[Dataset] Successfully loaded {len(df)} samples with labels.")
    return df

def get_transforms(mode="val"):
    """
    Standard preprocessing with conditional augmentation for training.
    """
    # Basic setup
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ]

    # Add Augmentations ONLY for training
    if mode == "train":
        transforms.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
            RandGaussianNoised(keys=["image"], prob=0.1),
        ])

    # Final scaling and resizing
    transforms.extend([
        Resized(keys=["image"], spatial_size=(224, 224, 80)),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True,
        ),
    ])

    return Compose(transforms)