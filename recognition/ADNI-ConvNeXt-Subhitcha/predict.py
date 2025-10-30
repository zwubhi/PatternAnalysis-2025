# predict.py
import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from modules import build_model

def load_classes(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def classify_image(image_path, model, classes, device, img_size):
    model.eval()
    tfm = get_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, idx = probs.max(1)
    return classes[idx.item()], conf.item(), img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best_model.pth")
    ap.add_argument("--classes", required=True, help="Path to classes.txt")
    ap.add_argument("--img", required=True, help="Path to image to classify")
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes)
    model = build_model(num_classes=len(classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    pred, conf, img = classify_image(args.img, model, classes, device, args.img_size)
    plt.imshow(img); plt.axis("off")
    plt.title(f"Predicted: {pred}\nConfidence: {conf*100:.2f}%")
    plt.show()

if __name__ == "__main__":
    main()
