import json
import pandas as pd
from dataset import load_odelia_metadata

def evaluate(json_path, data_root):
    print(f"\n--- Evaluating: {json_path} ---")
    
    with open(json_path, 'r') as f:
        preds = json.load(f)

    df = load_odelia_metadata(data_root)
    val_df = df[df["split"] == "val"].copy() 
    
    tp, fp, tn, fn = 0, 0, 0, 0
    correct = 0
    matches = 0

    for _, row in val_df.iterrows():
        pid = str(row['patient_id'])
        
        # INFER SIDE FROM IMAGE_PATH (Since columns 'side'/'laterality' are missing)
        path = str(row['image_path']).upper()
        side = 'L' if '_L' in path or '/L/' in path else 'R'

        actual_label = int(row['label'])
        
        # Match against your specific JSON structure
        if pid in preds and side in preds[pid]:
            matches += 1
            prob_dict = preds[pid][side]
            # Get class with highest probability
            pred_class_name = max(prob_dict, key=prob_dict.get)
            
            class_map = {"normal": 0, "benign": 1, "malignant": 2}
            pred_label = class_map[pred_class_name]

            # Binary evaluation: Malignant (2) vs. Non-Malignant (0, 1)
            if pred_label == 2 and actual_label == 2: tp += 1
            elif pred_label == 2 and actual_label != 2: fp += 1
            elif pred_label != 2 and actual_label != 2: tn += 1
            elif pred_label != 2 and actual_label == 2: fn += 1

            if pred_label == actual_label:
                correct += 1

    if matches == 0:
        print("Still no matches. Check if JSON keys match the PIDs in metadata.")
        return

    accuracy = correct / matches
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Matches found: {matches}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f} (Recall for Malignant)")
    print(f"Specificity: {specificity:.4f}")
    print(f"Metrics: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

data_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"
evaluate("runs/preds_densenet.json", data_path)
evaluate("runs/preds_mil.json", data_path)