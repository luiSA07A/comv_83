import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model

LABEL_NAMES = ["normal", "benign", "malignant"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model",      type=str, default="densenet",
                        choices=["densenet", "mil"])
    parser.add_argument("--subset",     type=str, default="val",
                        help="Which split to run inference on (val or RSH)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--in_channels",type=int, default=1)
    parser.add_argument("--output",     type=str, default="predictions.json")
    return parser.parse_args()

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = defaultdict(dict)

    for batch in loader:
        # 1. Handle Metadata Unpacking (Fix for ValueError/TypeError)
        images, labels, metadata = batch
        
        if isinstance(metadata, dict):
            patient_ids = metadata['patient_id']
            sides = metadata['laterality']
        else:
            patient_ids = metadata[0]
            sides = metadata[1]

        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 2. Match loop to actual Tensor size (Fix for IndexError)
        batch_count = images.size(0)

        for i in range(batch_count):
            pid = patient_ids[i]
            side = sides[i]
            prob = probs[i]
            
            results[pid][side] = {
                label: round(float(prob[j]), 6)
                for j, label in enumerate(LABEL_NAMES)
            }

    return dict(results)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 3. Use 'split' column (Fix for KeyError)
    df = load_odelia_metadata(args.data_root)
    test_df = df[df["split"] == args.subset].copy()
    print(f"[Test set] {len(test_df)} breast entries in subset '{args.subset}'")

    test_ds = OdeliaDataset(test_df, transform=get_transforms("test"))
    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # Load model
    model = get_model(args.model, in_channels=args.in_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    print(f"[Checkpoint] Loaded from {args.checkpoint}")

    predictions = predict(model, loader, device)

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[Output] Saved {len(predictions)} patient predictions to {args.output}")

if __name__ == "__main__":
    main()