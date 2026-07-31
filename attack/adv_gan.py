"""AdvGAN adversarial attack against a packet-level surrogate classifier.

The attack trains a Generator (G) to produce small perturbations on
individual packet feature vectors that fool the surrogate into predicting
the wrong class, while keeping the perturbation magnitude small.

Key design choices
------------------
- Input is a pre-extracted 1481-feature packet vector (CSV row) — no
  image manipulation is required.
- Only *mutable* feature positions are perturbed (controlled by the mask
  in :mod:`attack.masks`).  Immutable fields are left unchanged.
- Perturbation applied multiplicatively: ``perturbed = G(x) * mask * x + x``
- Loss = ``adv_lambda * BCE(surrogate(perturbed), 0) + pert_lambda * L2(G(x))``

Usage
-----
See :mod:`train_attack` for a full training script.
"""

import os

import torch
import torch.nn as nn

from models.generator import Generator
from attack.masks import build_perturbation_mask


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

    def train_batch(self, x: torch.Tensor, labels: torch.Tensor):
        """Run one optimisation step on a single mini-batch.

        Parameters
        ----------
        x : Tensor  (batch, 1481)
            Pre-extracted packet features (CSV row, normalised to [0, 1]).
        labels : Tensor  (batch,)
            Ground-truth class labels (unused in loss, kept for signature
            consistency).

        Returns
        -------
        tuple of floats
            ``(loss_perturb, loss_adv, gain)`` where *gain* is the number
            of packets that flipped from predicted-class-1 to predicted-class-0.
        """
        batch_size = x.shape[0]
        mask = build_perturbation_mask(n_features=x.shape[1]).to(self.device)

        # Generator expects (batch, 1, n_features)
        perturbation = self.G(x.unsqueeze(1)).squeeze(1)   # (batch, 1481)
        perturbation = perturbation * mask
        perturbed    = (perturbation * x + x).clamp(0.0, 1.0)

        self.optimizer_G.zero_grad()

        loss_perturb = torch.mean(
            torch.norm(perturbation, p=2, dim=1)
        )

        y_true        = torch.zeros(batch_size, 1, device=self.device)
        y_pred        = self.target_model(perturbed)
        y_pred_before = self.target_model(x)

        loss_adv = self._loss_fn(y_pred, y_true)

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
            Yields ``(pkts, labels)`` batches; pkts shape ``(B, 1481)``.
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

            for pkts, labels in train_dataloader:
                torch.cuda.empty_cache()
                pkts   = pkts.to(self.device)
                labels = labels.to(self.device)
                lp, la, gain = self.train_batch(pkts, labels)
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
