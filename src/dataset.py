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
    
    def get_paths(row):
        folder = os.path.join(data_root, "data", row['Institution'], "data_unilateral", row['uid'])
        pre = os.path.join(folder, "Pre.nii.gz")
        post = os.path.join(folder, "Post_1.nii.gz")
        if os.path.exists(pre) and os.path.exists(post):
            return pre, post
        return None, None

    df[['pre_path', 'post_path']] = df.apply(
        lambda r: pd.Series(get_paths(r)), axis=1
    )
    return df.dropna(subset=['pre_path'])

def get_transforms(mode="val"):
    return Compose([
        LoadImaged(keys=["pre", "post"]),
        EnsureChannelFirstd(keys=["pre", "post"]),
        # Normalize intensities individually before stacking
        ScaleIntensityRanged(keys=["pre", "post"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
        Orientationd(keys=["pre", "post"], axcodes="RAS"),
        Spacingd(keys=["pre", "post"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Resized(keys=["pre", "post"], spatial_size=(224, 224, 80)),
        # STACK THEM: Creates a 2-channel input [Pre, Post_1]
        ConcatItemsd(keys=["pre", "post"], name="image"), 
    ])