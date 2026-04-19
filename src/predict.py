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
        images, _, metadata = batch
        uids = metadata 
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for i in range(images.size(0)):
            uid = str(uids[i])
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
    parser.add_argument("--data_root", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/data_unilateral")
    parser.add_argument("--split_file", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/RSH/metadata_unilateral/split.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="densenet")
    parser.add_argument("--output_csv", type=str, default="predictions_leaderboard.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load RSH Specific Split
    df = pd.read_csv(args.split_file)
    df.columns = [c.lower() for c in df.columns] # Handle UID vs uid
    
    # 2. Build explicit paths to Post_1.nii.gz for every RSH folder
    if "image_path" not in df.columns:
        # Ensure we point to the absolute path of the Post_1 file
        df["image_path"] = df["uid"].apply(
            lambda x: os.path.join(args.data_root, x, "Post_1.nii.gz")
        )
    val_ds = OdeliaDataset(df, transform=get_transforms("val"))
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    raw_results = predict(model, loader, device)

    # 3. MAP TO THE CHALLENGE'S "WIDE" FORMAT
    # Get unique study IDs (AnonymizedXXX) and sort them alphabetically
    study_ids = sorted(list(set([u.replace('_left', '').replace('_right', '') for u in raw_results.keys()])))
    
    submission_rows = []
    for study in study_ids:
        # Map AnonymizedXXX to examID_N
        exam_id = f"examID_{study_ids.index(study) + 1}"
        
        # Get the scores for both sides
        # We use the 'malignant' probability as the score for the leaderboard
        left_uid = f"{study}_left"
        right_uid = f"{study}_right"
        
        # Pull the malignant score, default to 0.0 if the side is missing
        l_score = raw_results.get(left_uid, {}).get("left", {}).get("malignant", 0.0)
        r_score = raw_results.get(right_uid, {}).get("right", {}).get("malignant", 0.0)
        
        submission_rows.append({
            'studyID': exam_id,      # Matches their ground truth header
            'Lesion_Left': l_score,  # Matches their ground truth header
            'Lesion_Right': r_score  # Matches their ground truth header
        })

    # Save the wide-format CSV
    pd.DataFrame(submission_rows).to_csv(args.output_csv, index=False)
    print(f"[Done] Saved WIDE Format CSV to {args.output_csv}")

if __name__ == "__main__":
    main()