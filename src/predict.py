import json
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model
from collections import defaultdict

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = defaultdict(dict)

    for batch in loader:
        # Images, Labels, Metadata
        images, _, metadata = batch
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Extracting the full strings from the tuples
        # In a batch of 4, uids will be a tuple: ('ID1', 'ID2', 'ID3', 'ID4')
        uids = metadata[0] 
        sides = metadata[1]

        for i in range(images.size(0)):
            # Ensure we take the WHOLE string, not just the first char
            uid = str(uids[i])
            side = str(sides[i])
            prob = probs[i]
            
            # This ensures the JSON key is 'ODELIA_BRAID1_...' and not 'R'
            results[uid][side] = {
                "normal": round(float(prob[0]), 6),
                "benign": round(float(prob[1]), 6),
                "malignant": round(float(prob[2]), 6)
            }
    return dict(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="densenet")
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--subset", type=str, default="val") 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use your internal logic to get the correct validation dataframe
    df = load_odelia_metadata(args.data_root)
    val_df = df[df["split"] == "val"]
    
    val_ds = OdeliaDataset(val_df, transform=get_transforms("val"))
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    predictions = predict(model, loader, device)

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[Done] Saved predictions to {args.output}")

if __name__ == "__main__":
    main()