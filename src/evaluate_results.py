import json
import pandas as pd

def evaluate(json_path, csv_path):
    print(f"\n--- Debugging Matches for: {json_path} ---")
    
    with open(json_path, 'r') as f:
        preds = json.load(f)

    # Show the first key in your JSON to see its format
    sample_json_key = list(preds.keys())[0]
    print(f"Sample JSON Key: '{sample_json_key}'")

    df = pd.read_csv(csv_path)
    val_df = df[df["Split"] == "val"].copy() 
    
    # Show the first UID in your CSV
    sample_csv_uid = val_df['UID'].iloc[0]
    print(f"Sample CSV UID:  '{sample_csv_uid}'")

    tp, fp, tn, fn = 0, 0, 0, 0
    matches = 0

    for _, row in val_df.iterrows():
        uid = str(row['UID'])
        # Try to find which part of the UID is the key in your JSON
        # Your JSON likely has the Patient ID (e.g., '0158') 
        # while UID has the full string.
        
        match_key = None
        for k in preds.keys():
            if k == uid or k in uid: # Check if JSON key is inside the UID string
                match_key = k
                break
        
        if match_key:
            # Check for 'left'/'right' vs 'L'/'R'
            side_key = None
            for s in preds[match_key].keys():
                if s.lower() in uid.lower():
                    side_key = s
                    break

            if side_key:
                matches += 1
                # Logic for labels (Ensure 'label' column exists in your CSV)
                # If 'label' is missing from split_unilateral.csv, 
                # we must assume a placeholder or check your other metadata file.
                if 'label' in row:
                    actual = int(row['label'])
                    prob_dict = preds[match_key][side_key]
                    pred_class = max(prob_dict, key=prob_dict.get)
                    pred_label = {"normal": 0, "benign": 1, "malignant": 2}[pred_class]

                    if pred_label == 2 and actual == 2: tp += 1
                    elif pred_label == 2 and actual != 2: fp += 1
                    elif pred_label != 2 and actual != 2: tn += 1
                    elif pred_label != 2 and actual == 2: fn += 1

    print(f"FINAL RESULTS -> Matches: {matches} | TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

csv_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/split_unilateral.csv"
evaluate("runs/preds_densenet.json", csv_path)