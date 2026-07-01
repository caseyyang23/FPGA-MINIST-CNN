from __future__ import annotations

import argparse
from pathlib import Path

import torch

from quantize_export import build_quantized_bundle, load_checkpoint, quantized_forward_torch
from train_mnist import ROOT, make_loaders


def evaluate_quantized(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
    model = load_checkpoint(args.checkpoint, device)
    bundle = build_quantized_bundle(model, train_loader, device, args.calib_batches)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = quantized_forward_torch(images, bundle, device)
            labels = labels.to(device)
            correct += (pred == labels).sum().item()
            total += labels.numel()

    acc = correct / total
    print(f"quantized test accuracy: {acc * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the int8 quantized MNIST CNN.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "software" / "checkpoints" / "mnist_cnn_fp32.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "software" / "data")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_quantized(parse_args())
