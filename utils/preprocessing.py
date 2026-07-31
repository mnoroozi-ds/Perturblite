"""Preprocessing utilities for network-flow packet images.

The paper is **packet-based**: each individual packet is an independent
classification sample.  Flow images (stored as PNG/BMP on disk) are used
as the attack surface for the Generator but are exploded into per-packet
samples for training and evaluating the surrogate classifier.

Two key representations
-----------------------
1. **Generator input**  ``extract_generator_channel``
   Returns the raw forward (channel-0) flow image: ``(batch, 1, 15, 1501)``.
   Each of the 15 rows is one packet; the Generator perturbs all rows.

2. **Surrogate input**  ``flatten_for_surrogate``
   Takes a **single** packet row ``(batch, 1501)`` and strips
   routing-critical / per-connection header bytes, producing a flat
   ``(batch, 1481)`` feature vector.

   Kept column ranges (0-indexed, within each 1501-byte row)::

       col 0        : IP version + IHL       (1 col)
       cols 3–9     : DSCP/ECN … TTL         (7 cols)
       cols 26–36   : source/dest IP + ports (11 cols)
       cols 39–1500 : TCP/UDP payload        (1462 cols)

   Total: 1 + 7 + 11 + 1462 = 1481 features.

   Stripped (immutable / per-connection):
       1–2   : IP header checksum
       10–25 : IP options, TCP seq/ack, data-offset, flags
       37–38 : TCP checksum

3. **Flow→packets**  ``extract_flow_packets``
   Extracts all packet rows from a flow image and returns feature tensors
   together with a boolean validity mask (False for zero-padded rows).
"""

import torch


def extract_generator_channel(images: torch.Tensor) -> torch.Tensor:
    """Return the raw forward-direction (channel-0) slice for the Generator.

    Parameters
    ----------
    images : Tensor  (batch, 3, 15, 1501)

    Returns
    -------
    Tensor  (batch, 1, 15, 1501)
    """
    return images[:, 0:1, :, :]


def flatten_for_surrogate(pkt: torch.Tensor) -> torch.Tensor:
    """Strip immutable header bytes from a single packet row.

    Parameters
    ----------
    pkt : Tensor  (batch, 1501)
        Raw byte values of one packet per sample.

    Returns
    -------
    Tensor  (batch, 1481)
    """
    col0   = pkt[:, 0:1]       # 1 col
    cols3  = pkt[:, 3:10]      # 7 cols
    cols26 = pkt[:, 26:37]     # 11 cols
    cols39 = pkt[:, 39:1501]   # 1462 cols
    return torch.cat([col0, cols3, cols26, cols39], dim=1)


def extract_flow_packets(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-packet surrogate features from a batch of flow images.

    Processes all 15 packet rows in the forward channel (channel 0) and
    returns stripped feature vectors together with a validity mask that
    is ``False`` for zero-padded (absent) packet rows.

    Parameters
    ----------
    images : Tensor  (batch, C, 15, 1501)
        Raw flow image batch.

    Returns
    -------
    feats : Tensor  (batch, 15, 1481)
        Per-packet feature vectors.
    valid : Tensor  (batch, 15)  bool
        ``True`` where the corresponding packet row is non-zero.
    """
    fwd = images[:, 0, :, :]                    # (batch, 15, 1501)
    valid = fwd.abs().sum(dim=2) > 0            # (batch, 15)
    b, n, w = fwd.shape
    feats = flatten_for_surrogate(fwd.reshape(b * n, w))  # (batch*15, 1481)
    return feats.view(b, n, -1), valid


def flow_surrogate_prediction(
    classifier: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Compute a per-flow prediction by averaging over valid packet rows.

    For each flow in the batch, the surrogate is called on every non-zero
    packet.  The mean prediction across valid packets is returned as the
    flow-level score.

    Parameters
    ----------
    classifier : nn.Module
        Trained surrogate (packet-level binary classifier).
    images : Tensor  (batch, C, 15, 1501)
        Raw flow image batch.

    Returns
    -------
    Tensor  (batch, 1)  in [0, 1]
        Flow-level prediction (mean of per-packet predictions).
    """
    feats, valid = extract_flow_packets(images)   # (batch, 15, 1481), (batch, 15)
    b, n, f = feats.shape
    flat_preds = classifier(feats.view(b * n, f)) # (batch*15, 1)
    preds_2d   = flat_preds.view(b, n)            # (batch, 15)

    # Replace predictions for zero-padded rows with 0 before averaging
    preds_2d = preds_2d * valid.float()
    n_valid  = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
    return (preds_2d.sum(dim=1, keepdim=True) / n_valid)  # (batch, 1)


# ---------------------------------------------------------------------------
# Internal helper kept for the Generator’s adversarial image reconstruction
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
