"""Semantic feature masks and benign-distribution constraints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


DEFAULT_MUTABLE_PREFIXES = ("ip_header_", "tcp_header_")
PAYLOAD_PREFIXES = ("tcp_segment_data_", "payload_", "segment_data_")


def build_feature_masks(
    feature_names: Sequence[str],
    mutable_features: Iterable[str] | None = None,
    mutable_prefixes: Sequence[str] = DEFAULT_MUTABLE_PREFIXES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complementary immutable and mutable masks.

    PerturbLite changes retained TCP/IP header bytes and preserves segment
    data.  Named feature overrides support prepared data sets whose column
    naming differs from the standard ``ip_header_*``, ``tcp_header_*``, and
    ``tcp_segment_data_*`` convention.
    """
    lowered = [name.lower() for name in feature_names]
    explicit = set(mutable_features or ())
    unknown = explicit - set(feature_names)
    if unknown:
        raise ValueError(f"Mutable feature names not present in data: {sorted(unknown)}")
    protected = {
        name
        for name in explicit
        if name.lower().startswith(PAYLOAD_PREFIXES)
    }
    if protected:
        raise ValueError(
            "Payload/segment features are immutable in PerturbLite and cannot "
            f"be selected: {sorted(protected)}"
        )

    mutable = torch.zeros(len(feature_names), dtype=torch.float32)
    for index, (name, lower_name) in enumerate(zip(feature_names, lowered)):
        is_payload = lower_name.startswith(PAYLOAD_PREFIXES)
        selected = name in explicit or lower_name.startswith(tuple(mutable_prefixes))
        if selected and not is_payload:
            mutable[index] = 1.0

    if not torch.any(mutable):
        raise ValueError(
            "No mutable TCP/IP header features were identified. Use semantic "
            "column names (ip_header_*/tcp_header_*) or provide explicit mutable features."
        )

    immutable = 1.0 - mutable
    return immutable, mutable


def compute_benign_feature_bounds(
    dataloader,
    mutable_mask: torch.Tensor,
    upper_quantile: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``[benign minimum, benign q99]`` bounds for mutable features."""
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1]")

    benign_batches = []
    for features, labels in dataloader:
        benign = features[labels == 0]
        if benign.numel():
            benign_batches.append(benign.cpu())
    if not benign_batches:
        raise ValueError("Benign training samples are required to derive clipping bounds")

    benign_features = torch.cat(benign_batches, dim=0)
    lower = torch.zeros(benign_features.shape[1], dtype=torch.float32)
    upper = torch.ones(benign_features.shape[1], dtype=torch.float32)
    active = mutable_mask.bool().cpu()
    lower[active] = benign_features[:, active].amin(dim=0)
    upper[active] = torch.quantile(
        benign_features[:, active], upper_quantile, dim=0
    )
    # Constant benign features are valid and should remain constant.
    upper = torch.maximum(upper, lower)
    return lower, upper


def constrain_generated_sample(
    original: torch.Tensor,
    generated: torch.Tensor,
    immutable_mask: torch.Tensor,
    mutable_mask: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
) -> torch.Tensor:
    """Apply Algorithm 1: preserve immutable values and replace mutable ones."""
    generated_mutable = torch.maximum(
        torch.minimum(generated, upper_bounds), lower_bounds
    )
    return immutable_mask * original + mutable_mask * generated_mutable
