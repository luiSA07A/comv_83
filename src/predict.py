"""
TDT4265 - Inference Script
Generates predictions on the RSH test set in the format required
by the ODELIA leaderboard.

Usage:
  python predict.py \
    --data_root /datasets/tdt4265/ODELIA2025 \
    --checkpoint ./runs/densenet_xxx/best_model.pt \
    --model densenet \
    --output predictions.json
"""

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
    parser.add_argument("--subset",     type=str, default="RSH",
                        help="Which subset to run inference on")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--in_channels",type=int, default=1)
    parser.add_argument("--output",     type=str, default="predictions.json")
    return parser.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = defaultdict(dict)  # patient_id -> {left: probs, right: probs}

    for batch in loader:
        images, labels, metadata = batch
        
        # Extract metadata correctly from the batched object
        if isinstance(metadata, dict):
            # If DataLoader kept it as a dict of lists
            patient_ids = metadata['patient_id']
            sides = metadata['laterality']
        else:
            # If DataLoader turned it into a tuple (patient_ids_list, sides_list)
            patient_ids = metadata[0]
            sides = metadata[1]

        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for i in range(len(patient_ids)):
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

    # Load dataset (test/hidden subset)
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

    # Run inference
    predictions = predict(model, loader, device)

    # Save in leaderboard format
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[Output] Saved {len(predictions)} patient predictions to {args.output}")

    # Print a few examples
    print("\n--- Sample predictions ---")
    for pid, sides in list(predictions.items())[:3]:
        print(f"Patient {pid}:")
        for side, probs in sides.items():
            print(f"  {side}: {probs}")


if __name__ == "__main__":
    main()
