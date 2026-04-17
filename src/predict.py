import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model
from collections import defaultdict

LABEL_NAMES = ["normal", "benign", "malignant"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model",      type=str, default="densenet", choices=["densenet", "mil"])
    parser.add_argument("--subset",     type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output",     type=str, default="predictions.json")
    return parser.parse_args()

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = defaultdict(dict)

    for batch in loader:
        # FIX: The DataLoader returns (Image, Label, Patient_ID, Side)
        # In a batch, PIDs and Sides are returned as Tuples of Strings.
        images, labels, pids, sides = batch
        
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Iterate through the batch items using the actual batch size
        batch_count = images.size(0)
        for i in range(batch_count):
            # Ensure we are grabbing the full string from the tuple/list
            pid = str(pids[i]) 
            side = str(sides[i])
            prob = probs[i]
            
            # Use 'left' or 'right' if that's what is in your UID, 
            # or keep L/R if your dataset helper maps it that way.
            results[pid][side] = {
                label: round(float(prob[j]), 6)
                for j, label in enumerate(LABEL_NAMES)
            }

    return dict(results)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load data using the same unilateral split logic
    df = load_odelia_metadata(args.data_root)
    test_df = df[df["split"] == args.subset].copy()
    print(f"[Inference] Running on {len(test_df)} unilateral entries.")

    test_ds = OdeliaDataset(test_df, transform=get_transforms("test"))
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize and load model
    model = get_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)

    predictions = predict(model, loader, device)

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[Done] Saved full-ID predictions to {args.output}")

if __name__ == "__main__":
    main()