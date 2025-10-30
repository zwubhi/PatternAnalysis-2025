import torch
import torch.optim as optim
from tqdm import tqdm
from modules import ConvNeXt, LabelSmoothingLoss, cutmix_data
from dataset import get_data_loaders

def train_model(train_root, test_root, batch_size, epochs, device, seed, img_size=256):
    train_loader, val_loader, test_loader, classes = get_data_loaders(train_root, test_root, batch_size, seed, img_size)
    model = ConvNeXt(num_classes=len(classes)).to(device)

    criterion = LabelSmoothingLoss(classes=len(classes), smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val_acc = 0.0
    patience, no_improve = 3, 0
    train_acc_list, val_acc_list = [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if torch.rand(1) < 0.5:
                imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels)
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_list.append(train_acc)
        scheduler.step()
        print(f"Train Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / v_total
        val_acc_list.append(val_acc)
        print(f"Val Loss: {val_loss/v_total:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model, train_acc_list, val_acc_list, test_loader, device, classes

if __name__ == "__main__":
    import os
    import torch

    # Set your paths here
    BASE = "/content/drive/MyDrive" if os.path.exists("/content/drive/MyDrive") else "/content/drive/My Drive"
    train_root = f"{BASE}/PR_PROJ/ADNI/AD_NC/train"
    test_root = f"{BASE}/PR_PROJ/ADNI/AD_NC/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, train_acc, val_acc, test_loader, device, classes = train_model(
        train_root, test_root, batch_size=64, epochs=100, device=device, seed=1337, img_size=256)

    # You can later save train/val accuracy lists or plot them here
