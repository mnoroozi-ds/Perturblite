"""Preprocessing utilities for network-flow packet images.

Two representations are produced from the raw ``(batch, 3, 15, 1501)`` image:

1. **Generator input** ``extract_generator_channel`` → ``(batch, 1, 15, 1501)``
   The raw red (forward) channel, used as input to the Generator.

2. **Surrogate input** ``flatten_for_surrogate`` → ``(batch, 1481)``
   A flat feature vector derived from the *first packet row* of the forward
   channel after stripping routing-critical / per-connection header bytes:

   Kept column ranges (0-indexed, within each 1501-byte packet row)::

       col 0        : IP version + IHL       (1 col)
       cols 3–9     : DSCP/ECN … TTL         (7 cols)
       cols 26–36   : source/dest IP + ports (11 cols)
       cols 39–1500 : TCP/UDP payload        (1462 cols)

   Total: 1 + 7 + 11 + 1462 = 1481 features.

   Stripped columns (immutable / per-connection unique):
       1–2   : IP header checksum
       10–25 : IP options, TCP seq/ack, data-offset, flags
       37–38 : TCP checksum
"""

import torch


def extract_generator_channel(images: torch.Tensor) -> torch.Tensor:
    """Return the raw forward-direction (red) channel for the Generator.

    Parameters
    ----------
    images : Tensor  (batch, 3, 15, 1501)

    Returns
    -------
    Tensor  (batch, 1, 15, 1501)
    """
    return images[:, 0:1, :, :]


def flatten_for_surrogate(images: torch.Tensor, packet_row: int = 0) -> torch.Tensor:
    """Extract a flat 1481-feature vector from a flow image for the surrogate.

    Takes *packet_row* (default: first packet) from the forward (channel-0)
    direction and strips the immutable header bytes.

    Parameters
    ----------
    images : Tensor  (batch, 3, 15, 1501)
        Raw flow image.
    packet_row : int
        Which packet row to use (0 = first/forward packet).

    Returns
    -------
    Tensor  (batch, 1481)
    """
    # Forward channel, selected packet row  →  (batch, 1501)
    pkt = images[:, 0, packet_row, :]          # (batch, 1501)

    col0   = pkt[:, 0:1]                       # 1 col
    cols3  = pkt[:, 3:10]                      # 7 cols
    cols26 = pkt[:, 26:37]                     # 11 cols
    cols39 = pkt[:, 39:1501]                   # 1462 cols
    return torch.cat([col0, cols3, cols26, cols39], dim=1)  # (batch, 1481)


# ---------------------------------------------------------------------------
# Legacy helper — kept for the Generator's adversarial image reconstruction
# ---------------------------------------------------------------------------

def extract_classifier_channels(images: torch.Tensor) -> torch.Tensor:
    """Strip header bytes from all channels/rows (used inside AdvGAN).

    Parameters
    ----------
    images : Tensor  (batch, C, 15, 1501)

    Returns
    -------
    Tensor  (batch, C, 15, 1482)
    """
    col0   = images[:, :, :, 0:1]
    cols3  = images[:, :, :, 3:10]
    cols25 = images[:, :, :, 25:37]
    cols39 = images[:, :, :, 39:1501]
    return torch.cat([col0, cols3, cols25, cols39], dim=3)
