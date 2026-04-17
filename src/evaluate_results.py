import json
import pandas as pd

def evaluate(json_path, csv_path):
    print(f"\n--- Evaluating: {json_path} ---")
    
    with open(json_path, 'r') as f:
        preds = json.load(f)

    # Load the specific unilateral split file
    df = pd.read_csv(csv_path)
    # Filter for validation
    val_df = df[df["Split"] == "val"].copy() 
    
    tp, fp, tn, fn = 0, 0, 0, 0
    matches = 0

    for _, row in val_df.iterrows():
        uid = str(row['UID'])
        
        # 1. Parse the UID to get PID and Side
        # Example UID: ODELIA_BRAID1_0158_1_left
        parts = uid.split('_')
        # Patient ID is the part before the side (e.g., '0158')
        # Based on your JSON keys, let's try to find a match
        side = 'L' if 'left' in uid.lower() else 'R'
        
        # We need to find which part of the UID matches your JSON keys (like "R", "0", "2")
        # Let's check the number part of the UID
        pid_candidate = parts[2] if len(parts) > 2 else uid

        # 2. Flexible Matching against your JSON
        match_key = None
        for k in preds.keys():
            if k == pid_candidate or k in uid:
                match_key = k
                break

        # Assuming you have a 'label' column in this CSV
        if 'label' in row:
            actual_label = int(row['label'])
        else:
            # Fallback: You might need to merge this with your other metadata.csv 
            # if this file is only for splits.
            continue
        
        if match_key and side in preds[match_key]:
            matches += 1
            prob_dict = preds[match_key][side]
            pred_class = max(prob_dict, key=prob_dict.get)
            
            class_map = {"normal": 0, "benign": 1, "malignant": 2}
            pred_label = class_map[pred_class]

            if pred_label == 2 and actual_label == 2: tp += 1
            elif pred_label == 2 and actual_label != 2: fp += 1
            elif pred_label != 2 and actual_label != 2: tn += 1
            elif pred_label != 2 and actual_label == 2: fn += 1

    print(f"Matches: {matches} | TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

# Update path to use the new CSV
csv_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/split_unilateral.csv"
evaluate("runs/preds_densenet.json", csv_path)