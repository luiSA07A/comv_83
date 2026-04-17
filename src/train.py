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

def main():
    # Setup paths and device
    data_root = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data using official splits
    df = load_odelia_metadata(data_root)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    train_ds = OdeliaDataset(train_df, get_transforms("train"))
    val_ds = OdeliaDataset(val_df, get_transforms("val"))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    # Load chosen model (DenseNet or MIL)
    model = get_model("densenet").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    # Simplified loop for 10 epochs (Adjust as needed)
    for epoch in range(10):
        model.train()
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} complete.")

    total_hours = (time.time() - start_time) / 3600
    kwh = total_hours * 0.45 # Estimation for RTX 4090
    print(f"Sustainability: {kwh/16*100:.2f} km in a Tesla Model Y")

if __name__ == "__main__":
    main()