"""Train the PerturbLite surrogate classifier on packet-level CSV data."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import BinaryClassifier
from utils.dataset import build_packet_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PerturbLite surrogate DNN.")
    parser.add_argument("--data", required=True, help="Path to the prepared packet CSV.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--partition-col", default=None)
    parser.add_argument("--classifier-fraction", type=float, default=0.5)
    parser.add_argument("--attack-type-col", default=None)
    parser.add_argument("--attack-type", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--save-path", default="best_classifier.pth")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for packets, labels in dataloader:
        packets = packets.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        predictions = model(packets)
        batch_size = packets.shape[0]
        loss_sum += criterion(predictions, labels).item() * batch_size
        correct += ((predictions >= 0.5) == labels.bool()).sum().item()
        total += batch_size
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    loaders = build_packet_dataloaders(
        args.data,
        label_col=args.label_col,
        batch_size=args.batch_size,
        random_state=args.seed,
        partition="classifier",
        partition_col=args.partition_col,
        classifier_fraction=args.classifier_fraction,
        folds=args.folds,
        fold_index=args.fold_index,
        attack_type_col=args.attack_type_col,
        attack_type=args.attack_type,
    )
    model = BinaryClassifier(input_dim=len(loaders.feature_names)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(
        "Surrogate parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for packets, labels in loaders.train:
            packets = packets.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            predictions = model(packets)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * packets.shape[0]
            train_count += packets.shape[0]

        val_loss, val_accuracy = evaluate(model, loaders.validation, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss / max(train_count, 1):.6f} "
            f"| val_loss={val_loss:.6f} | val_accuracy={val_accuracy:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_names": loaders.feature_names,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                },
                args.save_path,
            )
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, loaders.test, criterion, device)
    print(
        f"Best validation loss: {best_loss:.6f} | test_loss={test_loss:.6f} "
        f"| test_accuracy={test_accuracy:.4f}"
    )


if __name__ == "__main__":
    train(parse_args())
