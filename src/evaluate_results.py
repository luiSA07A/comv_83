import json
import pandas as pd
from dataset import load_odelia_metadata

def evaluate(json_path, data_root):
    # 1. Load your results
    with open(json_path, 'r') as f:
        preds = json.load(f)
    
    df = load_odelia_metadata(data_root)
    val_df = df[df["split"] == "val"]

    print(f"Available columns: {val_df.columns.tolist()}")
    
    tp, fp, tn, fn = 0, 0, 0, 0
    correct = 0

    for _, row in val_df.iterrows():
        pid = str(row['patient_id'])
        side = row['side']
        actual_label = row['label'] # 0=normal, 1=benign, 2=malignant
        
        if pid in preds and side in preds[pid]:
            # Get the model's highest confidence class
            prob_dict = preds[pid][side]
            predicted_class = max(prob_dict, key=prob_dict.get)
            
            # Map class name back to number for comparison
            class_map = {"normal": 0, "benign": 1, "malignant": 2}
            pred_label = class_map[predicted_class]

            # Logic for Malignant (Label 2) detection
            if pred_label == 2 and actual_label == 2: tp += 1
            elif pred_label == 2 and actual_label != 2: fp += 1
            elif pred_label != 2 and actual_label != 2: tn += 1
            elif pred_label != 2 and actual_label == 2: fn += 1

            if pred_label == actual_label:
                correct += 1

    total = tp + fp + tn + fn
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n--- Metrics for {json_path} ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"TP: {tp} | FP: {fp}")
    print(f"TN: {tn} | FN: {fn}")
    print(f"Sensitivity (Recall): {tp/(tp+fn):.4f}" if (tp+fn)>0 else "Sens: N/A")

# Run it
evaluate("runs/preds_densenet.json", "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025")
evaluate("runs/preds_mil.json", "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025")