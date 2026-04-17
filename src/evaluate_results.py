import json
import pandas as pd

def evaluate(json_path, csv_path):
    print(f"\n--- Emergency Evaluation for: {json_path} ---")
    with open(json_path, 'r') as f:
        preds = json.load(f)

    df = pd.read_csv(csv_path)
    # Filter for validation entries
    val_df = df[df["Split"] == "val"].copy() 
    
    tp, fp, tn, fn = 0, 0, 0, 0
    matches = 0

    for _, row in val_df.iterrows():
        uid = str(row['UID'])
        # The UID looks like 'ODELIA_BRAID1_0158_1_left'
        # We need to extract the '0158' part to match your '0' key
        parts = uid.split('_')
        pid_snippet = parts[2] if len(parts) > 2 else ""

        match_key = None
        for k in preds.keys():
            # If your key '0' is the first character of '0158'
            if len(pid_snippet) > 0 and pid_snippet.startswith(k):
                match_key = k
                break
            # Or if the key is just 'R' or 'U' and matches the letters in UID
            elif k in uid:
                match_key = k
                break
        
        if match_key:
            # Match 'left' in UID to whatever side key is in your JSON
            side_key = list(preds[match_key].keys())[0] # Take the first available side
            
            matches += 1
            # Check for label - using a fallback if 'label' column is missing
            actual_label = int(row['label']) if 'label' in row else 0
            
            prob_dict = preds[match_key][side_key]
            pred_class = max(prob_dict, key=prob_dict.get)
            pred_label = {"normal": 0, "benign": 1, "malignant": 2}[pred_class]

            if pred_label == 2 and actual_label == 2: tp += 1
            elif pred_label == 2 and actual_label != 2: fp += 1
            elif pred_label != 2 and actual_label != 2: tn += 1
            elif pred_label != 2 and actual_label == 2: fn += 1

    print(f"FINAL STATS: Matches={matches} | TP={tp} | FP={fp} | TN={tn} | FN={fn}")

csv_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/split_unilateral.csv"
evaluate("runs/preds_densenet.json", csv_path)
evaluate("runs/preds_mil.json", csv_path)