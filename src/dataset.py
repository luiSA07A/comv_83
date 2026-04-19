import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose
import os

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