import time
import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model

def compute_metrics(labels, probs):
    # AUROC for malignant (label 2) vs others
    y_true = [1 if l == 2 else 0 for l in labels]
    y_score = [p[2] for p in probs]
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="ODELIA 2025 Training Script")
    parser.add_argument("--data_root", type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--model", type=str, default="densenet", help="densenet or mil")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. LOAD DATA AND CALCULATE WEIGHTS
    df = load_odelia_metadata(args.data_root)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    
    # Calculate weights based on label frequency (Normal, Benign, Malignant)
    counts = train_df["label"].value_counts().sort_index().to_list()
    class_weights = [1.0 / c for c in counts]
    
    # 2. CONSTRUCT BALANCED SAMPLER
    sample_weights = [class_weights[int(l)] for l in train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    print(f"[Status] Training on {len(train_df)} samples (Balanced), Validating on {len(val_df)} samples.")
    
    # 3. INITIALIZE DATASETS AND LOADERS
    # Uses "train" mode for augmentations and "val" for deterministic validation
    train_ds = OdeliaDataset(train_df, get_transforms("train"))
    val_ds = OdeliaDataset(val_df, get_transforms("val"))
    
    # Shuffle must be False when using a Sampler
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # 4. INITIALIZE MODEL AND WEIGHTED LOSS
    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Normalize weights so Normal class weight is 1.0
    loss_weights = torch.tensor([w / class_weights[0] for w in class_weights]).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    print(f"[Status] Using Weighted Loss: {loss_weights.tolist()}")

    # 5. TRAINING LOOP
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

        # SAVE CHECKPOINT
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, f"best_model_{args.model}.pt")
            torch.save(model.state_dict(), save_path)

    # 6. SUSTAINABILITY CALCULATION
    total_hours = (time.time() - start_time) / 3600
    kwh = total_hours * 0.45 
    tesla_km = (kwh / 16) * 100
    print(f"\n--- Sustainability Report ---")
    print(f"Energy used: {kwh:.2f} kWh")
    print(f"Sustainability: {tesla_km:.2f} km in a Tesla Model Y")

if __name__ == "__main__":
    main()