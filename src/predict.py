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
        # 1. Unpack the three items: Images, Labels, Metadata
        images, _, metadata = batch
        
        # 2. Extract IDs and Sides from the tuple
        # Based on OdeliaDataset, metadata[0] is usually PIDs/UIDs, metadata[1] is Sides
        # If your metadata is a dict, we handle that too.
        if isinstance(metadata, dict):
            uids = metadata.get('uid', metadata.get('patient_id'))
            sides = metadata.get('side', metadata.get('laterality'))
        else:
            # If it's a tuple/list: index 0 is ID, index 1 is Side
            uids = metadata[0]
            sides = metadata[1]
        
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for i in range(images.size(0)):
            uid = str(uids[i])
            side = str(sides[i])
            prob = probs[i]
            
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