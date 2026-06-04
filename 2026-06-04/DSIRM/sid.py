from __future__ import annotations

import torch


def prefix_match_features(query_codes: torch.Tensor, item_codes: torch.Tensor) -> torch.Tensor:
    """Hierarchical prefix matching features.

    query_codes, item_codes: (B, L)

    Returns:
        feats: (B, L) where feats[:, l] = 1 if prefix up to l matches else 0.
    """

    assert query_codes.shape == item_codes.shape

    bsz, levels = query_codes.shape
    feats = torch.zeros((bsz, levels), device=query_codes.device, dtype=torch.float32)

    prefix_ok = torch.ones((bsz,), device=query_codes.device, dtype=torch.bool)
    for l in range(levels):
        prefix_ok = prefix_ok & (query_codes[:, l] == item_codes[:, l])
        feats[:, l] = prefix_ok.float()

    return feats
