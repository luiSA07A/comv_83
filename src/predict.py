import json
import argparse
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import get_transforms, OdeliaDataset
from models import get_model
from collections import defaultdict

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = defaultdict(dict)
    for batch in loader:
        # metadata contains the UID (e.g., Anonymized100_left)
        images, _, metadata = batch
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for i in range(images.size(0)):
            uid = str(metadata[i])
            # Determine side from the UID string
            side = "left" if "left" in uid.lower() else "right"
            prob = probs[i]
            results[uid][side] = {
                "normal": round(float(prob[0]), 6),
                "benign": round(float(prob[1]), 6),
                "malignant": round(float(prob[2]), 6)
            }
    return dict(results)

def main():
    parser = argparse.ArgumentParser()
    # Defaults set for IDUN RSH paths
    parser.add_argument("--data_root", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/data_unilateral")
    parser.add_argument("--split_file", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/metadata_unilateral/split.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="densenet")
    parser.add_argument("--output_csv", type=str, default="predictions_leaderboard.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load RSH Specific Split
    df = pd.read_csv(args.split_file)
    df.columns = [c.lower() for c in df.columns] 
    
    # 2. Build explicit paths to Post_1.nii.gz for every RSH folder
    # This prevents the 'LoadImaged' RuntimeError by pointing to a file, not a folder
    df["image_path"] = df["uid"].apply(lambda x: os.path.join(args.data_root, x, "Post_1.nii.gz"))
    
    # Create Dataset and Loader
    val_ds = OdeliaDataset(df, transform=get_transforms("val"))
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=1)

    # Load Model
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Run Inference
    raw_results = predict(model, loader, device)

    # 3. MAP TO THE OFFICIAL LONG FORMAT
    submission_rows = []
    
    # Iterate through every UID we processed
    for uid, sides_dict in raw_results.items():
        # raw_results[uid] looks like {"left": {"normal": 0.9, ...}}
        # We just need the side name to get the values
        side = "left" if "left" in uid.lower() else "right"
        scores = sides_dict[side]
        
        # ENSURE SUM TO 1 (Floating point math can sometimes be off by 0.000001)
        total = scores['normal'] + scores['benign'] + scores['malignant']
        
        submission_rows.append({
            'ID': uid,  # Use the ORIGINAL UID (e.g., Anonymized100_left)
            'normal': scores['normal'] / total,
            'benign': scores['benign'] / total,
            'malignant': scores['malignant'] / total
        })

    # Save the CSV with the exact required headers
    df_final = pd.DataFrame(submission_rows)
    df_final.to_csv(args.output_csv, index=False)
    print(f"[Done] Saved Official Format CSV with {len(df_final)} rows.")

if __name__ == "__main__":
    main()