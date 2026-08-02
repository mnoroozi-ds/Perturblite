"""Evaluate a PerturbLite generator using attack success rate (ASR)."""

from __future__ import annotations

import argparse

import torch

from attack.masks import constrain_generated_sample
from models.classifier import BinaryClassifier
from models.generator import Generator
from utils.dataset import build_packet_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PerturbLite.")
    parser.add_argument("--data", required=True, help="Path to the prepared packet CSV.")
    parser.add_argument("--classifier-path", required=True)
    parser.add_argument("--generator-path", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--partition-col", default=None)
    parser.add_argument("--classifier-fraction", type=float, default=0.5)
    parser.add_argument("--attack-type-col", default=None)
    parser.add_argument("--attack-type", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_state(path: str, state_key: str, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint, checkpoint.get(state_key, checkpoint)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
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

    classifier_checkpoint, classifier_state = _load_state(
        args.classifier_path, "model_state_dict", device
    )
    saved_classifier_features = classifier_checkpoint.get("feature_names")
    if saved_classifier_features is not None and tuple(saved_classifier_features) != loaders.feature_names:
        raise ValueError("Classifier checkpoint feature order does not match the CSV")
    classifier = BinaryClassifier(input_dim=len(loaders.feature_names)).to(device)
    classifier.load_state_dict(classifier_state)
    classifier.eval()

    generator_checkpoint, generator_state = _load_state(
        args.generator_path, "generator_state_dict", device
    )
    required = {"feature_names", "immutable_mask", "mutable_mask", "lower_bounds", "upper_bounds"}
    missing = required - set(generator_checkpoint)
    if missing:
        raise ValueError(
            "Generator checkpoint does not include the required constraint metadata; "
            f"missing {sorted(missing)}"
        )
    if tuple(generator_checkpoint["feature_names"]) != loaders.feature_names:
        raise ValueError("Generator checkpoint feature order does not match the CSV")

    generator = Generator().to(device)
    generator.load_state_dict(generator_state)
    generator.eval()
    immutable_mask = generator_checkpoint["immutable_mask"].to(device)
    mutable_mask = generator_checkpoint["mutable_mask"].to(device)
    lower_bounds = generator_checkpoint["lower_bounds"].to(device)
    upper_bounds = generator_checkpoint["upper_bounds"].to(device)

    total = 0
    clean_correct = 0
    adversarial_correct = 0
    true_positive_attacks = 0
    successful_evasions = 0
    malicious_count = 0
    mutable_squared_error = 0.0
    mutable_count = int(mutable_mask.sum().item())

    with torch.no_grad():
        for packets, labels in loaders.test:
            packets = packets.to(device)
            labels = labels.to(device)
            malicious = labels == 1
            generated = generator(packets.unsqueeze(1)).squeeze(1)
            constrained = constrain_generated_sample(
                packets,
                generated,
                immutable_mask,
                mutable_mask,
                lower_bounds,
                upper_bounds,
            )
            # The generator targets malicious traffic only. Benign samples are
            # retained unchanged when reporting whole-split accuracy.
            crafted = torch.where(malicious.unsqueeze(1), constrained, packets)

            clean_labels = (classifier(packets).squeeze(1) >= 0.5).long()
            adversarial_labels = (classifier(crafted).squeeze(1) >= 0.5).long()
            eligible = malicious & (clean_labels == 1)

            total += labels.numel()
            clean_correct += (clean_labels == labels).sum().item()
            adversarial_correct += (adversarial_labels == labels).sum().item()
            true_positive_attacks += eligible.sum().item()
            successful_evasions += (eligible & (adversarial_labels == 0)).sum().item()
            malicious_count += malicious.sum().item()
            mutable_squared_error += (
                ((crafted[malicious] - packets[malicious]) * mutable_mask)
                .square()
                .sum()
                .item()
            )

    clean_accuracy = clean_correct / max(total, 1)
    adversarial_accuracy = adversarial_correct / max(total, 1)
    asr = successful_evasions / max(true_positive_attacks, 1)
    mse = mutable_squared_error / max(malicious_count * mutable_count, 1)

    print("\nPerturbLite evaluation")
    print(f"Total test packets: {total}")
    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Adversarial accuracy: {adversarial_accuracy:.4f}")
    print(
        "ASR (clean true-positive attacks classified benign): "
        f"{asr:.4f} ({successful_evasions}/{true_positive_attacks})"
    )
    print(f"Mutable-feature MSE on malicious packets: {mse:.6f}")


if __name__ == "__main__":
    main(parse_args())
