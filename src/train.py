import time
import torch
import torch.nn as nn
import argparse
import os
import math
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model

def compute_metrics(labels, probs):
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
    
    df = load_odelia_metadata(args.data_root)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    
    # Softer weights using sqrt
    counts = train_df["label"].value_counts().sort_index().to_list()
    class_weights = [1.0 / math.sqrt(c) for c in counts]  # ✅ sqrt instead of 1/c
    
    sample_weights = [class_weights[int(l)] for l in train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    print(f"[Status] Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")
    
    train_ds = OdeliaDataset(train_df, get_transforms("train"))
    val_ds = OdeliaDataset(val_df, get_transforms("val"))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_weights = torch.tensor([1.0 / math.sqrt(c) for c in counts])
    loss_weights = loss_weights / loss_weights[0]  # normalize to baseline
    loss_weights = loss_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"[Status] Softened Loss Weights: {loss_weights.tolist()}")

    start_time = time.time()
    best_auc = 0.0 
    patience = 7          # stop if no improvement for 7 epochs
    patience_counter = 0

    for epoch in range(args.epochs):
        # --- TRAINING ---
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
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}", flush=True)

        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(lbls.tolist())

        val_auc = compute_metrics(all_labels, all_probs)
        print(f"  Val AUC: {val_auc:.4f}", flush=True)

        if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0  # reset counter
        save_path = os.path.join(args.output_dir, f"best_model_{args.model}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"New best saved! AUC: {val_auc:.4f}", flush=True)
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{patience}", flush=True)
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break
        
        scheduler.step()  

    total_hours = (time.time() - start_time) / 3600
    kwh = total_hours * 0.35
    tesla_km = (kwh / 16) * 100
    print(f"\n--- Sustainability Report ---")
    print(f"Energy used: {kwh:.2f} kWh")
    print(f"Sustainability: {tesla_km:.2f} km in a Tesla Model Y")

if __name__ == "__main__":
    main()