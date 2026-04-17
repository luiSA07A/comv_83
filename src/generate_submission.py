import json
import pandas as pd
import argparse

def convert_json_to_csv(json_path, csv_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for uid, sides in data.items():
        for side, probs in sides.items():
            total = probs['normal'] + probs['benign'] + probs['malignant']
            
            rows.append({
                "ID": uid,
                "normal": probs['normal'] / total,
                "benign": probs['benign'] / total,
                "malignant": probs['malignant'] / total
            })

    df = pd.DataFrame(rows)
    df = df[["ID", "normal", "benign", "malignant"]]
    df.to_csv(csv_path, index=False)
    print(f"✅ Submission file created: {csv_path}")
    print(f"📊 Total rows: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="runs/preds_densenet.json")
    parser.add_argument("--output", type=str, default="predictions.csv")
    args = parser.parse_args()
    convert_json_to_csv(args.json, args.output)