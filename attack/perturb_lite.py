"""Constraint-aware PerturbLite generator training."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from attack.masks import constrain_generated_sample
from models.generator import Generator


class PerturbLite:
    """Train an attack-specific autoencoder against a frozen surrogate."""

    def __init__(
        self,
        device,
        target_model: nn.Module,
        feature_names: list[str] | tuple[str, ...],
        immutable_mask: torch.Tensor,
        mutable_mask: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        alpha: float = 0.95,
        beta: float = 0.05,
        lr: float = 1e-4,
        epsilon: float = 1e-7,
    ):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must both be positive")

        self.device = torch.device(device)
        self.target_model = target_model
        self.feature_names = tuple(feature_names)
        self.immutable_mask = immutable_mask.to(self.device)
        self.mutable_mask = mutable_mask.to(self.device)
        self.lower_bounds = lower_bounds.to(self.device)
        self.upper_bounds = upper_bounds.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.G = Generator().to(self.device)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.classification_normalizer = torch.tensor(1.0, device=self.device)
        self.perturbation_normalizer = torch.tensor(1.0, device=self.device)

        print(
            "Generator parameters: "
            f"{sum(p.numel() for p in self.G.parameters() if p.requires_grad):,}"
        )

    def craft(self, x: torch.Tensor) -> torch.Tensor:
        generated = self.G(x.unsqueeze(1)).squeeze(1)
        return constrain_generated_sample(
            original=x,
            generated=generated,
            immutable_mask=self.immutable_mask,
            mutable_mask=self.mutable_mask,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    def calibrate_normalizers(self, train_dataloader) -> None:
        """Derive attack-specific classification and perturbation normalizers."""
        max_classification = torch.tensor(0.0, device=self.device)
        self.target_model.eval()
        with torch.no_grad():
            for packets, _ in train_dataloader:
                packets = packets.to(self.device)
                probability = self.target_model(packets).clamp(
                    self.epsilon, 1.0 - self.epsilon
                )
                inverted_ce = -(
                    probability * torch.log(1.0 - probability)
                    + (1.0 - probability) * torch.log(probability)
                )
                max_classification = torch.maximum(
                    max_classification, inverted_ce.max()
                )

        mutable_ranges = (
            (self.upper_bounds - self.lower_bounds) * self.mutable_mask
        )
        max_perturbation = mutable_ranges.square().sum()
        self.classification_normalizer = max_classification.clamp_min(self.epsilon)
        self.perturbation_normalizer = max_perturbation.clamp_min(self.epsilon)
        print(
            "Loss normalizers - "
            f"classification: {self.classification_normalizer.item():.6f} | "
            f"perturbation: {self.perturbation_normalizer.item():.6f}"
        )

    def _losses(self, original: torch.Tensor, crafted: torch.Tensor):
        clean_probability = self.target_model(original).detach().clamp(
            self.epsilon, 1.0 - self.epsilon
        )
        crafted_probability = self.target_model(crafted).clamp(
            self.epsilon, 1.0 - self.epsilon
        )
        classification = -torch.mean(
            clean_probability * torch.log(1.0 - crafted_probability)
            + (1.0 - clean_probability) * torch.log(crafted_probability)
        ) / self.classification_normalizer
        perturbation = torch.mean(
            torch.sum((original - crafted).square(), dim=1)
        ) / self.perturbation_normalizer
        return classification, perturbation, clean_probability, crafted_probability

    def train_batch(self, packets: torch.Tensor) -> tuple[float, float, int]:
        self.optimizer_G.zero_grad()
        crafted = self.craft(packets)
        loss_classification, loss_perturbation, clean_p, crafted_p = self._losses(
            packets, crafted
        )
        loss = self.alpha * loss_classification + self.beta * loss_perturbation
        loss.backward()
        self.optimizer_G.step()

        successful = ((clean_p >= 0.5) & (crafted_p < 0.5)).sum().item()
        return loss_classification.item(), loss_perturbation.item(), successful

    @torch.no_grad()
    def validate(self, dataloader) -> tuple[float, float, float]:
        self.G.eval()
        total = 0
        success = 0
        classification_sum = 0.0
        perturbation_sum = 0.0
        for packets, _ in dataloader:
            packets = packets.to(self.device)
            crafted = self.craft(packets)
            loss_c, loss_p, clean_p, crafted_p = self._losses(packets, crafted)
            batch_size = packets.shape[0]
            classification_sum += loss_c.item() * batch_size
            perturbation_sum += loss_p.item() * batch_size
            eligible = clean_p >= 0.5
            success += (eligible & (crafted_p < 0.5)).sum().item()
            total += eligible.sum().item()
        return (
            classification_sum / max(len(dataloader.dataset), 1),
            perturbation_sum / max(len(dataloader.dataset), 1),
            success / max(total, 1),
        )

    def checkpoint(self, epoch: int | None = None) -> dict:
        return {
            "generator_state_dict": self.G.state_dict(),
            "epoch": epoch,
            "feature_names": self.feature_names,
            "immutable_mask": self.immutable_mask.detach().cpu(),
            "mutable_mask": self.mutable_mask.detach().cpu(),
            "lower_bounds": self.lower_bounds.detach().cpu(),
            "upper_bounds": self.upper_bounds.detach().cpu(),
            "classification_normalizer": self.classification_normalizer.detach().cpu(),
            "perturbation_normalizer": self.perturbation_normalizer.detach().cpu(),
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }

    def train(
        self,
        train_dataloader,
        validation_dataloader,
        epochs: int,
        checkpoint_every: int = 20,
        checkpoint_dir: str = "checkpoints",
        patience: int = 20,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad = False
        self.calibrate_normalizers(train_dataloader)

        best_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            self.G.train()
            classification_sum = 0.0
            perturbation_sum = 0.0
            successes = 0
            for packets, _ in train_dataloader:
                packets = packets.to(self.device)
                loss_c, loss_p, success = self.train_batch(packets)
                classification_sum += loss_c
                perturbation_sum += loss_p
                successes += success

            val_c, val_p, val_asr = self.validate(validation_dataloader)
            val_loss = self.alpha * val_c + self.beta * val_p
            batches = max(len(train_dataloader), 1)
            print(
                f"Epoch {epoch}/{epochs} | successes={successes} | "
                f"train_cls={classification_sum / batches:.6f} | "
                f"train_mse={perturbation_sum / batches:.6f} | "
                f"val_loss={val_loss:.6f} | val_asr={val_asr:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                stale_epochs = 0
                torch.save(
                    self.checkpoint(epoch),
                    Path(checkpoint_dir) / "G_best.pth",
                )
            else:
                stale_epochs += 1

            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                torch.save(
                    self.checkpoint(epoch),
                    Path(checkpoint_dir) / f"G_epoch_{epoch}.pth",
                )
            if patience > 0 and stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

        torch.save(self.checkpoint(epoch), Path(checkpoint_dir) / "G_final.pth")
        print(f"Saved generator checkpoints in {checkpoint_dir}")
