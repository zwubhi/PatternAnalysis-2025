# train.py
import os
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_datasets, get_loaders
from modules import build_model, LabelSmoothingLoss

# ---------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Folder with 'train' and 'test' subfolders")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--out_dir", type=str, default="./outputs")
    return p.parse_args()

# -------------- CutMix utils --------------
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

# -------------- Training --------------
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_root = os.path.join(args.data_dir, "train")
    test_root  = os.path.join(args.data_dir, "test")

    train_set, val_set, test_set, classes = get_datasets(train_root, test_root, args.img_size, args.seed)
    train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set, args.batch_size)

    # persist classes for predict/performance scripts
    with open(os.path.join(args.out_dir, "classes.txt"), "w") as f:
        for c in classes:
            f.write(c + "\n")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = LabelSmoothingLoss(classes=len(classes), smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    patience, no_improve, best_val_acc = 5, 0, 0.0
    train_loss_list, val_loss_list = [], []
    train_acc_list,  val_acc_list  = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        # ---- train ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if np.random.rand() < 0.5:
                imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels)
                with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc  = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # ---- validate ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=args.amp):
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # ---- early stopping + checkpoint ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save history for later plots
    np.savez(os.path.join(args.out_dir, "history.npz"),
             train_loss=train_loss_list, val_loss=val_loss_list,
             train_acc=train_acc_list,   val_acc=val_acc_list)

    # quick loss plot
    plt.figure(figsize=(8,5))
    xs = range(1, len(train_loss_list)+1)
    plt.plot(xs, train_loss_list, label="Training Loss")
    plt.plot(xs, val_loss_list,   label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"))
    plt.close()

    # test accuracy (final)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth"), map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"\nFinal Test Accuracy: {100.0 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
