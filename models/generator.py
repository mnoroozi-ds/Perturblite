"""PerturbLite autoencoder for generated packet feature values.

Architecture (1-D convolutional encoder–decoder)
------------------------------------------------
Processes one 1,481-feature packet as a 1-D byte sequence. The constraint
module selects generated mutable header values and copies immutable original
values; this network does not itself apply the masks.

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
    """1-D convolutional encoder-decoder with sigmoid feature output."""

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
        x : Tensor  ``(batch, 1, n_features)``
            Single packet as a 1-D sequence (1 channel, n_features values).

        Returns
        -------
        Tensor  ``(batch, 1, n_features)``
            Proposed feature values in ``[0, 1]``.
        """
        x = self.encoder(x)
        return self.decoder(x)
