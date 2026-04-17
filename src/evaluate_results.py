import json
import pandas as pd
import argparse
from dataset import load_odelia_metadata

def run_evaluation(json_path, data_root):
    df = load_odelia_metadata(data_root)
    val_df = df[df["split"] == "val"]
    
    with open(json_path, 'r') as f:
        preds = json.load(f)

    tp, fp, tn, fn = 0, 0, 0, 0
    matches = 0

    print(f"--- Evaluating {json_path} ---")

    for _, row in val_df.iterrows():
        # Match using the UID
        uid = str(row['uid']) if 'uid' in row else str(row['patient_id'])
        actual_label = int(row['label']) # 0=Normal, 1=Benign, 2=Malignant

        if uid in preds:
            # Get the side (assuming unilateral images have one side in the nested dict)
            side = list(preds[uid].keys())[0]
            prob_dict = preds[uid][side]
            
            # Get predicted class (the label with highest probability)
            pred_class_name = max(prob_dict, key=prob_dict.get)
            label_map = {"normal": 0, "benign": 1, "malignant": 2}
            pred_label = label_map[pred_class_name]

            matches += 1

            # Binary Metric: Malignant (2) vs. Non-Malignant (0, 1)
            if pred_label == 2 and actual_label == 2: tp += 1
            elif pred_label == 2 and actual_label != 2: fp += 1
            elif pred_label != 2 and actual_label != 2: tn += 1
            elif pred_label != 2 and actual_label == 2: fn += 1

    if matches > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Matches Found: {matches}")
        print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
        print(f"Sensitivity (Recall): {recall:.4f}")
        print(f"Precision: {precision:.4f}")
    else:
        print("Error: No matches found between JSON and Metadata. Check UID formats.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025")
    args = parser.parse_args()
    run_evaluation(args.json, args.data_root)