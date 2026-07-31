"""Generator network for adversarial perturbation of network flow images.

Architecture (1-D convolutional encoder–decoder)
------------------------------------------------
Processes each packet row independently as a 1-D byte sequence.
The forward pass reshapes ``(batch, 1, n_packets, n_bytes)`` →
``(batch * n_packets, 1, n_bytes)`` → Conv1d layers → reshape back.

Layer specification
-------------------
  Input                     channels = 1
  Hidden 1  Conv1d   BN ReLU   out = 32
  Hidden 2  Conv1d   BN ReLU   out = 64
  Hidden 3  Conv1d   BN ReLU   out = 128
  Hidden 4  ConvTr1d BN ReLU   out = 64
  Hidden 5  ConvTr1d BN ReLU   out = 32
  Output    ConvTr1d BN Sigmoid out = 1

Total learnable parameters: 61 953  (BN layers use affine=False)

Recommended hyperparameters
---------------------------
  Optimizer : Adam
  LR        : 1e-4
  Epochs    : 1000
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """1-D convolutional encoder–decoder generator.

    Input/output shape: ``(batch, 1, n_packets, n_bytes)``.
    Each packet row is processed independently through Conv1d layers so the
    generator is agnostic to the number of packets.
    """

    def __init__(self):
        super().__init__()

        # Encoder: expand channels 1 → 32 → 64 → 128
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=3, padding=1),
            nn.BatchNorm1d(32,  affine=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(32,  64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64,  affine=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128, affine=False),
            nn.ReLU(inplace=True),
        )

        # Decoder: collapse channels 128 → 64 → 32 → 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64,  affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64,  32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32,  affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32,  1,  kernel_size=3, padding=1),
            nn.BatchNorm1d(1,   affine=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor  ``(batch, 1, n_packets, n_bytes)``

        Returns
        -------
        Tensor  ``(batch, 1, n_packets, n_bytes)``
            Per-byte perturbation values in ``[0, 1]``.
        """
        batch, channels, n_packets, n_bytes = x.shape

        # Flatten packet dimension into the batch for 1-D convolutions
        # (batch, 1, n_packets, n_bytes) → (batch * n_packets, 1, n_bytes)
        x_1d = x.permute(0, 2, 1, 3).reshape(batch * n_packets, channels, n_bytes)

        x_1d = self.encoder(x_1d)
        x_1d = self.decoder(x_1d)

        # Restore original shape
        # (batch * n_packets, 1, n_bytes) → (batch, 1, n_packets, n_bytes)
        return x_1d.reshape(batch, n_packets, channels, n_bytes).permute(0, 2, 1, 3)
