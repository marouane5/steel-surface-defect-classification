"""Classify one steel image with a checkpoint produced by the notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


class SmallCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes)
        )

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))


def predict_defect(image_path, checkpoint_path):
    """Return six-class probabilities on CUDA using checkpoint preprocessing."""
    image_path, checkpoint_path = Path(image_path), Path(checkpoint_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required.")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    names = checkpoint["class_names"]
    if checkpoint["architecture"] == "small_cnn":
        model = SmallCNN(len(names))
    elif checkpoint["architecture"] == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(names))
    else:
        raise ValueError("Unsupported checkpoint architecture.")
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("L").convert("RGB")
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(transform(image).unsqueeze(0).to(device)).float(), dim=1
        )[0].cpu().tolist()
    best = max(range(len(names)), key=probabilities.__getitem__)
    return {
        "predicted_class": names[best],
        "confidence": probabilities[best],
        "class_probabilities": dict(zip(names, probabilities)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(predict_defect(args.image, args.checkpoint), indent=2))
