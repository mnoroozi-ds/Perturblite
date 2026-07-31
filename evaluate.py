"""Evaluate the AdvGAN attack success rate against the binary classifier.

Metrics reported
----------------
- **Attack Success Rate (ASR)**: fraction of originally-correct predictions
  that are flipped to the wrong class after perturbation.
- **Clean Accuracy**: classifier accuracy on unperturbed test images.
- **Adversarial Accuracy**: classifier accuracy after the generator perturbs
  the test images.

Usage
-----
    python evaluate.py \\
        --data-dir data/ \\
        --classifier-path best_classifier.pth \\
        --generator-path checkpoints/G_final.pth
"""

import argparse

import torch

from models.classifier import BinaryClassifier
from models.generator import Generator
from attack.masks import build_perturbation_mask
from utils.dataset import build_dataloaders
from utils.preprocessing import flow_surrogate_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the AdvGAN attack success rate."
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
        "--generator-path",
        required=True,
        help="Path to the trained Generator weights (.pth).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
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

    # Load models
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(
        torch.load(args.classifier_path, map_location=device)
    )
    classifier.eval()

    generator = Generator().to(device)
    generator.load_state_dict(
        torch.load(args.generator_path, map_location=device)
    )
    generator.eval()

    print(f"Loaded classifier  : {args.classifier_path}")
    print(f"Loaded generator   : {args.generator_path}")

    # Data — use test split
    _, test_loader = build_dataloaders(args.data_dir, batch_size=args.batch_size)

    # Perturbation mask
    mask = build_perturbation_mask(n_packets=15, n_bytes=1501).to(device)

    # Evaluation loop
    #
    # ASR is defined as:
    #
    #   ASR = T_m / TP
    #
    # where:
    #   TP  = number of samples correctly classified on clean (unperturbed) input
    #   T_m = number of those TP samples that the adversarial perturbation causes
    #         the model to misclassify  (i.e., originally-correct → now-wrong)
    #
    # Reference: equation (3) in the paper.

    total   = 0
    TP      = 0   # correctly classified on clean input
    T_m     = 0   # TP samples successfully flipped by the attack
    adv_correct = 0  # correctly classified on adversarial input (for reporting)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            batch_size = images.size(0)

            # --- clean predictions (mean over valid packets per flow) ---
            clean_preds = flow_surrogate_prediction(classifier, images)
            clean_pred_labels = (clean_preds >= 0.5).int()
            label_ints = labels.int()
            correctly_classified = (clean_pred_labels == label_ints)   # bool mask

            # --- generate adversarial perturbation ---
            channel_red = images[:, 0:1, :, :]
            perturbation = generator(channel_red)
            expanded_mask = mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)
            perturbation = perturbation * expanded_mask
            perturbed_red = (perturbation * channel_red + channel_red).clamp(0, 1)

            adv_images = torch.cat(
                [perturbed_red, images[:, 1:3, :, :]], dim=1
            )

            # --- adversarial predictions (mean over valid packets per flow) ---
            adv_preds = flow_surrogate_prediction(classifier, adv_images)
            adv_pred_labels = (adv_preds >= 0.5).int()

            # --- accumulate counters ---
            TP          += correctly_classified.sum().item()
            # T_m: was correct on clean, now wrong on adversarial
            T_m         += (correctly_classified & (adv_pred_labels != label_ints)).sum().item()
            adv_correct += (adv_pred_labels == label_ints).sum().item()
            total       += batch_size

    clean_acc = TP          / total if total > 0 else 0.0
    adv_acc   = adv_correct / total if total > 0 else 0.0
    asr       = T_m         / TP    if TP    > 0 else 0.0  # ASR = T_m / TP

    print(f"\n{'=' * 40}")
    print(f"Total test samples  : {total}")
    print(f"TP  (clean correct) : {TP}  →  clean accuracy = {clean_acc:.4f}")
    print(f"T_m (TP flipped)    : {T_m}")
    print(f"Adversarial accuracy: {adv_acc:.4f}  ({adv_correct}/{total})")
    print(f"ASR = T_m / TP      : {asr:.4f}  ({T_m}/{TP})")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main(parse_args())
