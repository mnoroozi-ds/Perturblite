"""Byte-position masks that define which packet features may be perturbed.

Only *mutable* fields — fields whose value does not affect packet validity
or network routing — should be perturbed so that generated adversarial
packets remain structurally valid.

Mutable positions in the 1481-feature vector
---------------------------------------------
These indices correspond to the following original packet bytes (after the
data generation pipeline strips immutable header bytes):

  Feature index  7  ← original byte  9 : IP TTL
  Feature index 17  ← original byte 35 : TCP urgent pointer (high byte)
  Feature index 18  ← original byte 36 : TCP urgent pointer (low byte)

If your feature extraction pipeline uses a different column order, update
``MUTABLE_FEATURE_POSITIONS`` to match.
"""

import torch


# Feature indices (in the 1481-dim vector) that are safe to perturb.
MUTABLE_FEATURE_POSITIONS: list[int] = [7, 17, 18]


def build_perturbation_mask(n_features: int = 1481) -> torch.Tensor:
    """Return a 1-D float mask of shape ``(n_features,)``.

    Entries are 1.0 at mutable feature positions and 0.0 elsewhere.

    Parameters
    ----------
    n_features : int
        Total number of features per packet (default: 1481).

    Returns
    -------
    torch.Tensor  shape ``(n_features,)``
    """
    mask = torch.zeros(n_features)
    for idx in MUTABLE_FEATURE_POSITIONS:
        if idx < n_features:
            mask[idx] = 1.0
    return mask
