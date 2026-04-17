import time
import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model

def compute_metrics(labels, probs):
    # AUROC for malignant (label 2) vs others
    y_true = [1 if l == 2 else 0 for l in labels]
    y_score = [p[2] for p in probs]
    return roc_auc_score(y_true, y_score)

def main():
    # 1. Setup Argument Parser to match your .sh scripts
    parser = argparse.ArgumentParser(description="ODELIA 2025 Training Script")
    parser.add_argument("--data_root", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="densenet", help="densenet or mil")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./runs")
    args = parser.parse_args()

    # 2. Setup Device and Directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load data using official splits and your new path logic
    df = load_odelia_metadata(args.data_root)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    print(f"[Status] Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")
    
    train_ds = OdeliaDataset(train_df, get_transforms("train"))
    val_ds = OdeliaDataset(val_df, get_transforms("val"))
    
    # Added num_workers to speed up 3D data loading
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # 4. Initialize Model, Optimizer, and Loss
    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg Loss: {avg_loss:.4f}", flush=True)

        # 6. Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, f"best_model_{args.model}.pt")
            torch.save(model.state_dict(), save_path)

    # 7. Sustainability Calculation (The Tesla km)
    total_hours = (time.time() - start_time) / 3600
    kwh = total_hours * 0.45 # Constant for RTX 4090/A100 power draw
    tesla_km = (kwh / 16) * 100
    print(f"\n--- Sustainability Report ---")
    print(f"Energy used: {kwh:.2f} kWh")
    print(f"Sustainability: {tesla_km:.2f} km in a Tesla Model Y")

if __name__ == "__main__":
    main()