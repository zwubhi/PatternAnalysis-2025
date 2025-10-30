import torch
from modules import ConvNeXt
import matplotlib.pyplot as plt

def test_model(model_path, test_loader, device, classes):
    model = ConvNeXt(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    import os
    import torch
    from dataset import get_data_loaders

    BASE = "/content/drive/MyDrive" if os.path.exists("/content/drive/MyDrive") else "/content/drive/My Drive"
    train_root = f"{BASE}/PR_PROJ/ADNI/AD_NC/train"
    test_root = f"{BASE}/PR_PROJ/ADNI/AD_NC/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data only
    _, _, test_loader, classes = get_data_loaders(train_root, test_root, batch_size=64, seed=1337, img_size=256)

    test_model("best_model.pth", test_loader, device, classes)
