import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

def get_data_loaders(train_root, test_root, batch_size, seed, img_size=256):
    train_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    val_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    base_dataset = datasets.ImageFolder(train_root)
    y_all = np.array([lbl for _, lbl in base_dataset.samples])
    idx_all = np.arange(len(y_all))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr_idx, va_idx = next(sss.split(idx_all, y_all))

    train_set = Subset(datasets.ImageFolder(train_root, transform=train_tfm), tr_idx.tolist())
    val_set = Subset(datasets.ImageFolder(train_root, transform=val_tfm), va_idx.tolist())
    test_set = datasets.ImageFolder(test_root, transform=val_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    classes = base_dataset.classes

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(f"Classes: {classes}")

    return train_loader, val_loader, test_loader, classes
