import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model

def compute_metrics(labels, probs):
    # AUROC for malignant (label 2) vs others
    y_true = [1 if l == 2 else 0 for l in labels]
    y_score = [p[2] for p in probs]
    return roc_auc_score(y_true, y_score)

import argparse # Add this at the top with your other imports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="densenet")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_root = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = load_odelia_metadata(data_root)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    train_ds = OdeliaDataset(train_df, get_transforms("train"))
    val_ds = OdeliaDataset(val_df, get_transforms("val"))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)

    # Use the model name from the argument
    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    
    # Use args.epochs instead of a hardcoded 10
    for epoch in range(args.epochs):
        model.train()
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        
        # Recommendation: Add a validation check here to track AUROC!
        print(f"Epoch {epoch+1}/{args.epochs} complete.")

    total_hours = (time.time() - start_time) / 3600
    kwh = total_hours * 0.45 
    print(f"Sustainability: {kwh/16*100:.2f} km in a Tesla Model Y")
    
    # Save the model so you can use it for the leaderboard!
    torch.save(model.state_code(), f"best_model_{args.model}.pt")

if __name__ == "__main__":
    main()