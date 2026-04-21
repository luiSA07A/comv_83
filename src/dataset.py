import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, Resized, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandZoomd, RandGaussianNoised
)
import os
import glob


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
                print(f"Error loading image at {img_path}: {e}")
                raise e

        label = int(row["label"]) if "label" in row.index else -1
        uid = str(row['uid'])
        
        return data["image"], label, uid

def load_odelia_metadata(data_root):
    split_file = os.path.join(data_root, "split_unilateral.csv")
    df_split = pd.read_csv(split_file)
    
    all_annotations = []
    for inst in df_split['Institution'].unique():
        anno_path = os.path.join(data_root, "data", inst, "metadata_unilateral", "annotation.csv")
        if os.path.exists(anno_path):
            df_anno = pd.read_csv(anno_path)
            all_annotations.append(df_anno)
            
    df_all_anno = pd.concat(all_annotations).drop_duplicates(subset=['UID'])
    
    df = pd.merge(df_split, df_all_anno[['UID', 'Lesion']], on='UID', how='left')

    def fill_missing_label(row):
        if pd.isna(row['Lesion']):
            # get the opposite side UID
            if '_right' in row['UID']:
                counterpart = row['UID'].replace('_right', '_left')
            else:
                counterpart = row['UID'].replace('_left', '_right')
            match = df_all_anno[df_all_anno['UID'] == counterpart]['Lesion']
            if len(match) > 0:
                return match.values[0]
            return None  # still unknown, drop later
        return row['Lesion']
    
    df['Lesion'] = df.apply(fill_missing_label, axis=1)
    
    df = df.rename(columns={'UID': 'uid', 'Split': 'split', 'Lesion': 'label'})
    df['split'] = df['split'].str.lower()
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    def get_path(row):
        folder_path = os.path.join(data_root, "data", row['Institution'], "data_unilateral", row['uid'])
        matches = glob.glob(os.path.join(folder_path, "Post_1.nii.gz"))
        if matches:
            return matches[0]
        return None

    df['image_path'] = df.apply(get_path, axis=1)
    
    initial_len = len(df)
    df = df.dropna(subset=['image_path'])
    print(f"[Dataset] Found {len(df)} valid images (Dropped {initial_len - len(df)} missing files)")
    print(f"[Dataset] Label distribution:\n{df['label'].value_counts()}")
    print(f"[Dataset] Train/val split:\n{df['split'].value_counts()}")
    
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