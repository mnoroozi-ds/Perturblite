"""Train the AdvGAN adversarial perturbation generator.

Usage
-----
    python train_attack.py --data-dir data/ --classifier-path best_classifier.pth --epochs 200

The generator learns to produce small, targeted perturbations that fool the
pre-trained binary classifier into misclassifying network flows.

Only mutable byte positions (TTL, TCP urgent pointer, first option byte) are
perturbed, ensuring generated packets remain structurally valid.
"""

import argparse

import torch

from models.classifier import BinaryClassifier
from attack.adv_gan import AdvGAN
from utils.dataset import build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AdvGAN generator against the flow classifier."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory containing sub-folders '0' and '1'.",
    )
    parser.add_argument(
        "--classifier-path",
        required=True,
        help="Path to the pre-trained classifier weights (.pth).",
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
        help="Learning rate for the Generator optimiser (default: 1e-4).",
    )
    parser.add_argument(
        "--adv-lambda",
        type=float,
        default=10.0,
        help="Weight for adversarial loss term (default: 10.0).",
    )
    parser.add_argument(
        "--pert-lambda",
        type=float,
        default=0.0,
        help="Weight for perturbation magnitude loss term (default: 0.0).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save a checkpoint every N epochs (default: 20).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cuda', 'cpu', or leave empty for auto-detect.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load frozen classifier
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(
        torch.load(args.classifier_path, map_location=device)
    )
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    print(f"Loaded classifier from {args.classifier_path}")

    # Data
    train_loader, _ = build_dataloaders(args.data_dir, batch_size=args.batch_size)

    # Attack
    attacker = AdvGAN(
        device=device,
        target_model=classifier,
        adv_lambda=args.adv_lambda,
        pert_lambda=args.pert_lambda,
        lr=args.lr,
    )

    attacker.train(
        train_dataloader=train_loader,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main(parse_args())
