"""
TDT4265 - Training & Evaluation Pipeline
Supports both DenseNet and MIL models.
Run locally or on IDUN — just set DATA_ROOT and MODEL_NAME.
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import load_odelia_metadata, get_transforms, OdeliaDataset
from models import get_model


# ─── Config ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, required=True,
                        help="Path to ODELIA2025 dataset root")
    parser.add_argument("--model",      type=str, default="densenet",
                        choices=["densenet", "mil"])
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--in_channels",type=int, default=1,
                        help="1=single phase, 5=pre+post1-4, 6=+T2")
    parser.add_argument("--spatial_size", nargs=3, type=int, default=[96, 96, 32])
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_auroc(labels, probs):
    """
    Compute malignant-vs-non-malignant AUROC as required by ODELIA challenge.
    labels: list of ints (0=normal, 1=benign, 2=malignant)
    probs:  list of softmax arrays shape (3,)
    """
    y_true = [1 if l == 2 else 0 for l in labels]
    y_score = [p[2] for p in probs]   # malignant probability
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_operating_point_metrics(labels, probs, target_sensitivity=0.90):
    """
    Specificity at 90% sensitivity, and Sensitivity at 90% specificity.
    """
    from sklearn.metrics import roc_curve
    y_true = np.array([1 if l == 2 else 0 for l in labels])
    y_score = np.array([p[2] for p in probs])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr

    # Specificity @ 90% sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    spec_at_sens = specificity[idx]

    # Sensitivity @ 90% specificity
    idx2 = np.argmin(np.abs(specificity - target_sensitivity))
    sens_at_spec = tpr[idx2]

    return {
        f"specificity_at_{int(target_sensitivity*100)}sens": float(spec_at_sens),
        f"sensitivity_at_{int(target_sensitivity*100)}spec": float(sens_at_spec),
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in loader:
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(all_labels)
    auroc = compute_auroc(all_labels, all_probs)
    op_metrics = compute_operating_point_metrics(all_labels, all_probs)

    return {
        "loss": total_loss / n,
        "auroc": auroc,
        **op_metrics,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) / f"{args.model}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] Output dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── Data ──
    df = load_odelia_metadata(args.data_root)
    # Exclude hidden test set (RSH) from training
    df_train_val = df[df["subset"] != "RSH"].copy()

    train_df, val_df = train_test_split(
        df_train_val, test_size=0.15, stratify=df_train_val["label_int"],
        random_state=args.seed
    )
    print(f"[Split] Train: {len(train_df)}  Val: {len(val_df)}")

    spatial_size = tuple(args.spatial_size)
    train_ds = OdeliaDataset(train_df, transform=get_transforms("train", spatial_size))
    val_ds   = OdeliaDataset(val_df,   transform=get_transforms("val",   spatial_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    model = get_model(args.model, in_channels=args.in_channels).to(device)
    print(f"[Model] {args.model} — params: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted loss to handle imbalance
    class_counts = train_df["label_int"].value_counts().sort_index().values
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_auroc = 0.0
    history = []
    total_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        total_train_time += elapsed

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()},
            "epoch_time_s": round(elapsed, 1),
        }
        history.append(row)
        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.3f} "
              f"| val_auroc {val_metrics['auroc']:.4f} | {elapsed:.0f}s")

        # Save best checkpoint
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ New best AUROC: {best_auroc:.4f}")

    # Save training history
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    # Sustainability note
    total_hours = total_train_time / 3600
    # RTX 4090 ≈ 450W TDP; Tesla Model Y uses ~16 kWh/100km
    kwh = total_hours * 0.45
    km_equivalent = (kwh / 16) * 100
    print(f"\n[Sustainability] Total training time: {total_hours:.2f}h")
    print(f"  Estimated energy use: {kwh:.2f} kWh")
    print(f"  Equivalent Tesla Model Y range: {km_equivalent:.1f} km")
    print(f"  (Trondheim → Oslo is ~495 km)")

    with open(output_dir / "sustainability.json", "w") as f:
        json.dump({
            "total_train_hours": round(total_hours, 2),
            "estimated_kwh": round(kwh, 2),
            "tesla_model_y_km": round(km_equivalent, 1),
        }, f, indent=2)

    print(f"\n[Done] Best val AUROC: {best_auroc:.4f}. Results in {output_dir}")


if __name__ == "__main__":
    main()
