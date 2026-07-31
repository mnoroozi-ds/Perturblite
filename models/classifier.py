"""Surrogate (binary) classifier for network flows.

Input shape: (batch, input_dim)  — flat feature vector per flow.
Default input_dim = 1481 features derived from a single forward-direction
packet after stripping routing-critical header bytes (IP checksum, TCP
checksum, sequence/ack numbers).  See ``utils.preprocessing.flatten_for_surrogate``
for the extraction logic.

Output: scalar in [0, 1] via Sigmoid (class 1 probability).

Architecture
------------
  Input   → 1481
  Hidden1 → FC(2048) + Dropout(0.1) + ReLU
  Hidden2 → FC(1024) + Dropout(0.1) + ReLU
  Hidden3 → FC(512)  + Dropout(0.1) + ReLU
  Hidden4 → FC(256)  + Dropout(0.1) + ReLU
  Hidden5 → FC(64)   + Dropout(0.1) + ReLU
  Output  → FC(1)    + Sigmoid

Recommended hyperparameters
---------------------------
  Optimizer : Adam
  LR        : 1e-4
  Epochs    : 200
"""

import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """Five-hidden-layer MLP surrogate classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features per sample (default: 1481).
    """

    def __init__(self, input_dim: int = 1481):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),

            nn.Linear(256, 64),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor  (batch, input_dim)
            Pre-flattened feature vector.  Use
            ``utils.preprocessing.flatten_for_surrogate`` to convert raw
            flow images to this representation.

        Returns
        -------
        Tensor  (batch, 1)  in [0, 1]
        """
        return self.net(x)
