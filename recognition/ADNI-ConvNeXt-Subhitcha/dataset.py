# dataset.py
import os
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_transforms(img_size: int):
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    return train_tfm, val_tfm

def get_datasets(train_root: str, test_root: str, img_size: int, seed: int):
    train_tfm, val_tfm = get_transforms(img_size)

    base_dataset = datasets.ImageFolder(train_root)  # no transforms here (for stratified split)
    y_all = np.array([lbl for _, lbl in base_dataset.samples])
    idx_all = np.arange(len(y_all))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr_idx, va_idx = next(sss.split(idx_all, y_all))

    train_set = Subset(datasets.ImageFolder(train_root, transform=train_tfm), tr_idx.tolist())
    val_set   = Subset(datasets.ImageFolder(train_root, transform=val_tfm), va_idx.tolist())
    test_set  = datasets.ImageFolder(test_root, transform=val_tfm)

    classes: List[str] = base_dataset.classes
    return train_set, val_set, test_set, classes

def get_loaders(train_set, val_set, test_set, batch_size: int, num_workers: int = 4, pin_memory: bool = True):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
