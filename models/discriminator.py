"""Discriminator network (optional — not used in default advGAN training).

Classifies 3-channel flow images as real or adversarial.
Input shape: (batch, 3, 15, W)
Output shape: (batch, W) — soft assignment via Softmax.
"""

import torch.nn as nn


class Discriminator(nn.Module):
    """Three-layer convolutional discriminator.

    Uses stride-2 convolutions instead of pooling for downsampling.
    The classifier head outputs a per-column probability distribution
    (Softmax over W dimension) rather than a scalar, matching the
    spatial resolution of flow images.
    """

    def __init__(self, output_dim: int = 1486):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(47616, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
