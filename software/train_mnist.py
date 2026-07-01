from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyMNISTCNN(nn.Module):
    """Small CNN matched to the HLS accelerator template."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32, bias=True)
        self.fc2 = nn.Linear(32, 10, bias=True)

    def forward(self, x: torch.Tensor, return_layers: bool = False):
        conv1_relu = F.relu(self.conv1(x))
        pool1 = self.pool(conv1_relu)
        conv2_relu = F.relu(self.conv2(pool1))
        pool2 = self.pool(conv2_relu)
        flat = pool2.flatten(1)
        fc1_relu = F.relu(self.fc1(flat))
        logits = self.fc2(fc1_relu)

        if not return_layers:
            return logits

        return logits, {
            "conv1_relu": conv1_relu,
            "pool1": pool1,
            "conv2_relu": conv2_relu,
            "pool2": pool2,
            "fc1_relu": fc1_relu,
            "logits": logits,
        }


def make_loaders(batch_size: int, data_dir: Path, num_workers: int = 0):
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()

    return correct / total


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    model = TinyMNISTCNN().to(device)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"model parameters: {param_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.numel()
            total += labels.numel()

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / total

        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={avg_loss:.4f} "
            f"train_acc={train_acc * 100:.2f}% "
            f"test_acc={test_acc * 100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "test_accuracy": best_acc,
                    "epoch": epoch,
                    "arch": "conv1(1,8,3), conv2(8,16,3), fc1(784,32), fc2(32,10)",
                    "input_transform": "ToTensor only, no normalization",
                },
                args.output,
            )
            print(f"saved best checkpoint: {args.output}, acc={best_acc * 100:.2f}%")

    history_path = args.output.parent / "train_history.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"saved history: {history_path}")
    print(f"best test accuracy: {best_acc * 100:.2f}%")

    if best_acc >= 0.98:
        print("PASS: float32 accuracy >= 98%")
    else:
        print("WARNING: accuracy < 98%, try --epochs 10 or --lr 0.001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny MNIST CNN for FPGA HLS export.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "software" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "software" / "checkpoints" / "mnist_cnn_fp32.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
