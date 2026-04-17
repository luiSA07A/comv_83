import json
import pandas as pd
from dataset import load_odelia_metadata

def evaluate(json_path, data_root):
    print(f"\n--- Evaluating: {json_path} ---")
    
    # 1. Load results
    try:
        with open(json_path, 'r') as f:
            preds = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return

    # 2. Load actual labels
    df = load_odelia_metadata(data_root)
    # Filter for validation set (David, ensure your train.py used 'val')
    val_df = df[df["split"] == "val"].copy() 
    
    # DEBUG: See what the columns actually are
    cols = val_df.columns.tolist()
    print(f"Detected columns: {cols}")

    # Determine which column name to use for the side/laterality
    side_col = 'laterality' if 'laterality' in cols else ('side' if 'side' in cols else None)
    if not side_col:
        print("Error: Could not find 'laterality' or 'side' in columns!")
        return

    tp, fp, tn, fn = 0, 0, 0, 0
    correct = 0

    for _, row in val_df.iterrows():
        pid = str(row['patient_id'])
        side = str(row[side_col]) # Use the dynamically found column name
        actual_label = int(row['label']) # 0=normal, 1=benign, 2=malignant
        
        # Match against JSON structure (e.g., preds[patient_id][side])
        if pid in preds and side in preds[pid]:
            prob_dict = preds[pid][side]
            predicted_class_name = max(prob_dict, key=prob_dict.get)
            
            class_map = {"normal": 0, "benign": 1, "malignant": 2}
            pred_label = class_map[predicted_class_name]

            # Binary evaluation: Malignant (2) vs. Non-Malignant (0, 1)
            if pred_label == 2 and actual_label == 2: tp += 1
            elif pred_label == 2 and actual_label != 2: fp += 1
            elif pred_label != 2 and actual_label != 2: tn += 1
            elif pred_label != 2 and actual_label == 2: fn += 1

            if pred_label == actual_label:
                correct += 1

    total = tp + fp + tn + fn
    if total == 0:
        print("No matches found between metadata and predictions. Check Patient IDs.")
        return

    accuracy = correct / len(val_df)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f} (Recall for Malignant)")
    print(f"Specificity: {specificity:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Running both
data_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"
evaluate("runs/preds_densenet.json", data_path)
evaluate("runs/preds_mil.json", data_path)