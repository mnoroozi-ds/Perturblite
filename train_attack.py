"""Train an attack-specific PerturbLite generator."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from attack.perturb_lite import PerturbLite
from attack.masks import build_feature_masks, compute_benign_feature_bounds
from models.classifier import BinaryClassifier
from utils.dataset import build_packet_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PerturbLite generator.")
    parser.add_argument("--data", required=True, help="Path to the prepared packet CSV.")
    parser.add_argument("--classifier-path", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--partition-col", default=None)
    parser.add_argument("--classifier-fraction", type=float, default=0.5)
    parser.add_argument("--attack-type-col", default=None)
    parser.add_argument("--attack-type", default=None)
    parser.add_argument(
        "--mutable-features",
        default=None,
        help="Comma-separated feature names; defaults to ip_header_*/tcp_header_* columns.",
    )
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def malicious_loader(loader: DataLoader, batch_size: int, shuffle: bool) -> DataLoader:
    labels = loader.dataset.y
    indices = torch.nonzero(labels == 1, as_tuple=False).flatten().tolist()
    if not indices:
        raise ValueError("Generator training requires malicious packets (label 1)")
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
    )


def load_classifier(path: str, feature_names, device) -> BinaryClassifier:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    saved_features = checkpoint.get("feature_names") if isinstance(checkpoint, dict) else None
    if saved_features is not None and tuple(saved_features) != tuple(feature_names):
        raise ValueError("Classifier checkpoint feature order does not match the CSV")
    classifier = BinaryClassifier(input_dim=len(feature_names)).to(device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    return classifier


def main(args: argparse.Namespace) -> None:
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
        partition="generator",
        partition_col=args.partition_col,
        classifier_fraction=args.classifier_fraction,
        folds=args.folds,
        fold_index=args.fold_index,
        attack_type_col=args.attack_type_col,
        attack_type=args.attack_type,
    )
    classifier = load_classifier(args.classifier_path, loaders.feature_names, device)

    explicit_mutable = None
    if args.mutable_features:
        explicit_mutable = [name.strip() for name in args.mutable_features.split(",") if name.strip()]
    immutable_mask, mutable_mask = build_feature_masks(
        loaders.feature_names,
        mutable_features=explicit_mutable,
    )
    lower_bounds, upper_bounds = compute_benign_feature_bounds(
        loaders.train,
        mutable_mask,
        upper_quantile=args.upper_quantile,
    )
    print(
        f"Mutable header features: {int(mutable_mask.sum().item())} | "
        f"immutable features: {int(immutable_mask.sum().item())}"
    )

    attacker = PerturbLite(
        device=device,
        target_model=classifier,
        feature_names=loaders.feature_names,
        immutable_mask=immutable_mask,
        mutable_mask=mutable_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        alpha=args.alpha,
        beta=args.beta,
        lr=args.lr,
    )
    attacker.train(
        train_dataloader=malicious_loader(loaders.train, args.batch_size, shuffle=True),
        validation_dataloader=malicious_loader(
            loaders.validation, args.batch_size, shuffle=False
        ),
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )


if __name__ == "__main__":
    main(parse_args())
