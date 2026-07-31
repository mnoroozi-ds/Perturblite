"""Train the binary network-flow classifier.

Usage
-----
    python train_classifier.py --data-dir data/ --epochs 1000 --save-path best_classifier.pth

The data directory must have the structure::

    data/
      0/   <- benign flow images
      1/   <- malicious flow images

All images are split 90/10 into train/test by default.
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import BinaryClassifier
from utils.dataset import build_packet_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BinaryClassifier on network-flow images."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory containing sub-folders '0' and '1'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--save-path",
        default="best_classifier.pth",
        help="Path to save the best model checkpoint (default: best_classifier.pth).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cuda', 'cpu', or leave empty for auto-detect.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # build_packet_dataloaders splits at flow level then explodes to packets
    train_loader, test_loader = build_packet_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    model = BinaryClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Surrogate parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- training pass ---
        model.train()
        train_loss = 0.0
        for pkts, labels in train_loader:
            # pkts: (batch, 1481) pre-extracted features
            feats  = pkts.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_loader)

        # --- evaluation pass ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for pkts, labels in test_loader:
                feats  = pkts.to(device)   # (batch, 1481) already extracted
                labels = labels.float().unsqueeze(1).to(device)
                preds = model(feats)
                test_loss += criterion(preds, labels).item()
                predicted = (preds >= 0.5).int()
                correct += (predicted == labels.int()).sum().item()
                total += labels.size(0)

        mean_test_loss = test_loss / len(test_loader)
        accuracy = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={mean_train_loss:.4f} | "
            f"test_loss={mean_test_loss:.4f} | "
            f"accuracy={accuracy:.4f}"
        )

        if mean_test_loss < best_loss:
            best_loss = mean_test_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model → {args.save_path}")

    print(f"Training complete. Best test loss: {best_loss:.4f}")


if __name__ == "__main__":
    train(parse_args())
