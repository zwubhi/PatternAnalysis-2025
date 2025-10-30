import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from modules import build_model   # Import ConvNeXt model constructor

def load_classes(path: str):
    """
    Load class label names from a text file, one class per line.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_transform(img_size):
    """
    Construct the preprocessing pipeline for input images.
    This should match the pipeline used at training time.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def classify_image(image_path, model, classes, device, img_size):
    """
    Preprocess the image, run it through the model, and return predicted class and confidence.

    Args:
        image_path: Path to input image.
        model: Trained PyTorch model.
        classes: List of class names.
        device: CPU or CUDA.
        img_size: Size to resize input.

    Returns:
        predicted class name, confidence (float), image (PIL)
    """
    model.eval()
    tfm = get_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, idx = probs.max(1) 
    return classes[idx.item()], conf.item(), img

def main():
    ap = argparse.ArgumentParser(description="Classify a single image using ConvNeXt")
    ap.add_argument("--weights", required=True, help="Path to ConvNeXt.pth (model weights)")
    ap.add_argument("--classes", required=True, help="Path to classes.txt (class names)")
    ap.add_argument("--img", required=True, help="Path to image to classify")
    ap.add_argument("--img_size", type=int, default=256, help="Input image size for resizing")
    args = ap.parse_args()

    # Set computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load class names from file
    classes = load_classes(args.classes)
    # Build and load model
    model = build_model(num_classes=len(classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # Run prediction and display
    pred, conf, img = classify_image(args.img, model, classes, device, args.img_size)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {pred}\nConfidence: {conf*100:.2f}%")
    plt.show()

if __name__ == "__main__":
    main()
