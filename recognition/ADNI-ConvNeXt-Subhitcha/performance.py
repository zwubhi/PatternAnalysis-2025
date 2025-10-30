# performance.py
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, RocCurveDisplay,
                             ConfusionMatrixDisplay)

from dataset import get_datasets, get_loaders
from modules import build_model

def load_classes(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with 'train' and 'test'")
    ap.add_argument("--weights", required=True, help="Path to best_model.pth")
    ap.add_argument("--classes", required=True, help="Path to classes.txt")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--history", type=str, default=None, help="Optional history.npz for curves")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes)

    train_root = os.path.join(args.data_dir, "train")
    test_root  = os.path.join(args.data_dir, "test")
    _, _, test_set, _ = get_datasets(train_root, test_root, args.img_size, seed=1337)
    _, _, test_loader = get_loaders(None, None, test_set, args.batch_size)  # only need test loader

    model = build_model(num_classes=len(classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    # ---- collect predictions ----
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y = np.array(all_labels)
    p = np.array(all_probs)
    yhat = np.array(all_preds)

    # ---- metrics ----
    roc_auc   = roc_auc_score(y, p)
    cm        = confusion_matrix(y, yhat)
    acc       = accuracy_score(y, yhat)
    prec      = precision_score(y, yhat)
    rec       = recall_score(y, yhat)
    tn, fp, fn, tp = cm.ravel()
    spec      = tn / (tn + fp)
    f1        = f1_score(y, yhat)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- plots ----
    plt.figure(figsize=(6,6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png")); plt.close()

    plt.figure(figsize=(6,6))
    RocCurveDisplay.from_predictions(y, p)
    plt.title("ROC Curve"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png")); plt.close()

    # Optional: training/validation curves if history.npz exists
    if args.history and os.path.exists(args.history):
        h = np.load(args.history, allow_pickle=True)
        train_loss = h["train_loss"]; val_loss = h["val_loss"]
        train_acc  = h["train_acc"];  val_acc  = h["val_acc"]
        xs = range(1, len(train_loss)+1)

        plt.figure(figsize=(8,5))
        plt.plot(xs, train_loss, label="Training Loss")
        plt.plot(xs, val_loss,   label="Validation Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "loss_curve.png")); plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(xs, train_acc, label="Training Accuracy")
        plt.plot(xs, val_acc,   label="Validation Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "accuracy_curve.png")); plt.close()

if __name__ == "__main__":
    main()
