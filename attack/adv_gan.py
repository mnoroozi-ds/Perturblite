"""AdvGAN adversarial attack against a network-flow binary classifier.

The attack trains a Generator (G) to produce small perturbations on
network-flow packet images that fool the target classifier into predicting
the wrong class, while keeping the perturbation magnitude small.

Key design choices
------------------
- Only *mutable* packet byte positions are perturbed (controlled by a mask
  defined in :mod:`attack.masks`).  Immutable fields (e.g. IP/TCP source
  port, sequence numbers) are left unchanged.
- Perturbation is applied multiplicatively: ``perturbed = G(x) * mask * x + x``
  This scales the perturbation relative to the original byte value, producing
  more realistic modifications.
- Loss = ``adv_lambda * BCE(classifier(perturbed), 0) + pert_lambda * L2(G(x))``
  The adversarial term drives misclassification; the perturbation term limits
  the size of the change.

Usage
-----
See :mod:`train_attack` for a full training script.
"""

import os

import torch
import torch.nn as nn

from models.generator import Generator
from attack.masks import build_perturbation_mask
from utils.preprocessing import extract_classifier_channels, flatten_for_surrogate


class AdvGAN:
    """GAN-based adversarial attack.

    Parameters
    ----------
    device : str or torch.device
        Compute device, e.g. ``"cuda"`` or ``"cpu"``.
    target_model : nn.Module
        Pre-trained binary classifier to attack (kept frozen).
    adv_lambda : float
        Weight for the adversarial (misclassification) loss term.
    pert_lambda : float
        Weight for the perturbation magnitude loss term.
    lr : float
        Learning rate for the Generator optimiser.
    """

    def __init__(
        self,
        device,
        target_model: nn.Module,
        adv_lambda: float = 10.0,
        pert_lambda: float = 0.0,
        lr: float = 1e-4,
    ):
        self.device = device
        self.target_model = target_model
        self.adv_lambda = adv_lambda
        self.pert_lambda = pert_lambda

        self.G = Generator().to(device)
        print(f"Generator parameters: {sum(p.numel() for p in self.G.parameters() if p.requires_grad):,}")

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr)
        self._loss_fn = nn.BCELoss()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_batch(self, x: torch.Tensor, imagereal: torch.Tensor, epoch: int):
        """Run one optimisation step on a single mini-batch.

        Parameters
        ----------
        x : Tensor  (batch, 1, 15, W)
            Single-channel (red) slice of the flow image — input to G.
        imagereal : Tensor  (batch, 3, 15, 1501)
            Full 3-channel flow image — used to build the classifier input.
        epoch : int
            Current epoch number (used for debug logging).

        Returns
        -------
        tuple of floats
            ``(loss_perturb, loss_adv, gain)`` where *gain* is the number
            of samples that flipped from predicted-class-1 to predicted-class-0.
        """
        batch_size = imagereal.shape[0]

        # --- build surrogate-ready flat features ---
        image_real_real = extract_classifier_channels(imagereal)  # kept for adv reconstruction
        feat_real = flatten_for_surrogate(imagereal)               # (batch, 1481) → surrogate

        # --- build perturbation mask (only mutable byte positions) ---
        mask = build_perturbation_mask(n_packets=15, n_bytes=1501).to(self.device)
        expanded_mask = mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)

        # --- generate & apply perturbation ---
        perturbation = self.G(x)
        perturbation = perturbation * expanded_mask
        perturbed = perturbation * x + x

        # --- rebuild 3-channel adversarial image for the classifier ---
        red = extract_classifier_channels(
            torch.cat([
                perturbed,
                imagereal[:, 1:3, :, :],
            ], dim=1)
        )
        red = torch.clamp(red, 0.0, 1.0)

        green = image_real_real[:, 1, :, :].unsqueeze(1)
        blue  = image_real_real[:, 2, :, :].unsqueeze(1)
        adv_image = torch.cat([red, green, blue], dim=1)

        # --- Generator loss ---
        self.optimizer_G.zero_grad()

        loss_perturb = torch.mean(
            torch.norm(perturbation.view(batch_size, -1), p=2, dim=1)
        )

        y_true = torch.zeros(batch_size, 1, device=self.device)
        feat_adv = flatten_for_surrogate(
            torch.cat([adv_image[:, 0:1, :, :], imagereal[:, 1:3, :, :]], dim=1)
        )
        y_pred = self.target_model(feat_adv)
        y_pred_before = self.target_model(feat_real)

        loss_adv = self._loss_fn(y_pred, y_true)

        # count samples that flipped toward the target class
        gain = torch.sum(
            (y_pred < 0.5).int() > (y_pred_before < 0.5).int()
        ).item()

        loss_G = self.adv_lambda * loss_adv + self.pert_lambda * loss_perturb
        loss_G.backward()
        self.optimizer_G.step()

        return loss_perturb.item(), loss_adv.item(), gain

    def train(self, train_dataloader, epochs: int, checkpoint_every: int = 20):
        """Train the Generator for *epochs* epochs.

        Checkpoints are saved every *checkpoint_every* epochs as
        ``checkpoints/G_epoch_<N>.pth``.  A final ``checkpoints/G_final.pth``
        is saved after the last epoch.

        Parameters
        ----------
        train_dataloader : DataLoader
            Yields ``(images, labels)`` batches; images shape (B, 3, 15, 1501).
        epochs : int
            Number of full passes over the training set.
        checkpoint_every : int
            Save a checkpoint every this many epochs.
        """
        os.makedirs("checkpoints", exist_ok=True)
        self.target_model.eval()

        for epoch in range(epochs):
            loss_perturb_sum = 0.0
            loss_adv_sum = 0.0
            gain_sum = 0

            for images, labels in train_dataloader:
                torch.cuda.empty_cache()
                images = images.to(self.device)
                # red channel only → input to G
                channel_red = images[:, 0:1, :, :]
                lp, la, gain = self.train_batch(channel_red, images, epoch)
                loss_perturb_sum += lp
                loss_adv_sum     += la
                gain_sum         += gain

            n = len(train_dataloader)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"gain={gain_sum} | "
                f"loss_perturb={loss_perturb_sum / n:.4f} | "
                f"loss_adv={loss_adv_sum / n:.4f}"
            )

            if (epoch + 1) % checkpoint_every == 0:
                path = f"checkpoints/G_epoch_{epoch + 1}.pth"
                torch.save(self.G.state_dict(), path)
                print(f"  Saved checkpoint: {path}")

        torch.save(self.G.state_dict(), "checkpoints/G_final.pth")
        print("Saved final generator: checkpoints/G_final.pth")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_weights(self, g_path: str) -> None:
        """Load Generator weights from *g_path*."""
        self.G.load_state_dict(torch.load(g_path, map_location=self.device))
        print(f"Loaded Generator weights from {g_path}")
