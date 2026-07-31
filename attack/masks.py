"""Byte-position masks that define which packet fields may be perturbed.

Only *mutable* fields — fields whose value does not affect packet validity
or network routing — should be perturbed so that generated adversarial
packets remain structurally valid.

Mutable columns (0-indexed byte positions within each packet row):
  - 9  : IP TTL field
  - 35 : TCP urgent pointer (low byte)
  - 36 : TCP options / padding first byte

All other byte positions are treated as immutable (mask value = 0).
"""

import torch


# Byte positions (column indices) that are safe to perturb.
MUTABLE_BYTE_POSITIONS: list[int] = [9, 35, 36]


def build_perturbation_mask(n_packets: int = 15, n_bytes: int = 1501) -> torch.Tensor:
    """Return a float mask of shape ``(n_packets, n_bytes)``.

    Entries are 1.0 at mutable byte positions and 0.0 elsewhere.

    Parameters
    ----------
    n_packets : int
        Number of packet rows (height dimension of flow image).
    n_bytes : int
        Number of byte columns (width dimension of flow image).

    Returns
    -------
    torch.Tensor
        Boolean mask as a float tensor, shape ``(n_packets, n_bytes)``.
    """
    mask = torch.zeros(n_packets, n_bytes)
    for col in MUTABLE_BYTE_POSITIONS:
        if col < n_bytes:
            mask[:, col] = 1.0
    return mask
