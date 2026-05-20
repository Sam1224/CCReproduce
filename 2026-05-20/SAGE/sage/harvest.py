from __future__ import annotations

from dataclasses import dataclass

import torch

from .gates import KNNDensityGate, MahalanobisGate
from .simhash import SimHash


@dataclass(frozen=True)
class HarvestStats:
    n_unlabeled: int
    n_candidates: int
    n_harvested: int
    pass_maha: int
    pass_knn: int


def floor_stratified_sample(
    codes: torch.Tensor,
    *,
    n_samples: int,
    floor_per_bucket: int,
    seed: int,
) -> torch.Tensor:
    """Floor-constrained sampling over SimHash buckets.

    For each bucket we sample at least floor_per_bucket examples (if available), then
    fill the remaining budget proportionally to bucket size.
    """

    rng = torch.Generator().manual_seed(seed)
    unique, inverse, counts = torch.unique(codes, return_inverse=True, return_counts=True)

    bucket_indices: list[torch.Tensor] = []
    remaining_budget = n_samples

    # First pass: floors
    for bucket_id in range(unique.numel()):
        idx = torch.nonzero(inverse == bucket_id, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        k = min(floor_per_bucket, idx.numel())
        perm = idx[torch.randperm(idx.numel(), generator=rng)[:k]]
        bucket_indices.append(perm)
        remaining_budget -= perm.numel()

    if remaining_budget <= 0:
        return torch.cat(bucket_indices, dim=0)[:n_samples]

    # Second pass: proportional fill
    probs = counts.float() / counts.sum().float()
    extra_per_bucket = torch.floor(probs * remaining_budget).to(torch.int64)

    extras: list[torch.Tensor] = []
    for bucket_id in range(unique.numel()):
        idx = torch.nonzero(inverse == bucket_id, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        already = 0
        for t in bucket_indices:
            if t.numel() > 0 and codes[t[0]] == unique[bucket_id]:
                already = t.numel()
                break
        # allow resampling for simplicity in toy (no de-dup strictness)
        k = int(extra_per_bucket[bucket_id].item())
        if k <= 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=rng)[: min(k, idx.numel())]]
        extras.append(perm)

    all_idx = torch.cat(bucket_indices + extras, dim=0)
    if all_idx.numel() > n_samples:
        all_idx = all_idx[torch.randperm(all_idx.numel(), generator=rng)[:n_samples]]
    return all_idx


@torch.no_grad()
def head_bucket_sample(
    x_unlabeled: torch.Tensor,
    *,
    n_samples: int,
    simhash_bits: int = 16,
    head_coverage: float = 0.8,
    seed: int = 0,
) -> torch.Tensor:
    """A naive baseline sampler that over-focuses on head buckets.

    This mimics the representation-bias issue described in the paper: rare cohorts may be missed.

    Returns indices (into x_unlabeled).
    """

    if not (0.0 < head_coverage <= 1.0):
        raise ValueError("head_coverage must be in (0, 1]")

    x_unlabeled = x_unlabeled.float()
    n_unlabeled, n_features = x_unlabeled.shape
    simhash = SimHash(n_bits=simhash_bits, n_features=n_features, seed=seed)
    codes = simhash.codes(x_unlabeled)

    unique, inverse, counts = torch.unique(codes, return_inverse=True, return_counts=True)
    order = torch.argsort(counts, descending=True)

    cum = torch.cumsum(counts[order], dim=0).float() / counts.sum().float()
    keep = order[cum <= head_coverage]
    if keep.numel() == 0:
        keep = order[:1]

    keep_mask = torch.isin(inverse, keep)
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    rng = torch.Generator().manual_seed(seed)
    if keep_idx.numel() <= n_samples:
        return keep_idx
    return keep_idx[torch.randperm(keep_idx.numel(), generator=rng)[:n_samples]]


@torch.no_grad()
def harvest_confident_negatives(
    x_unlabeled: torch.Tensor,
    *,
    candidate_size: int = 8000,
    simhash_bits: int = 16,
    floor_per_bucket: int = 10,
    post_floor_per_bucket: int = 5,
    target_harvest_size: int | None = None,
    maha_quantile: float = 0.95,
    knn_quantile: float = 0.2,
    vote_threshold: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, HarvestStats]:
    """Toy SAGE negative harvesting: SimHash stratify + gating ensemble.

    Returns indices (into x_unlabeled) of harvested confident negatives.
    """

    x_unlabeled = x_unlabeled.float()
    n_unlabeled, n_features = x_unlabeled.shape

    simhash = SimHash(n_bits=simhash_bits, n_features=n_features, seed=seed)
    codes = simhash.codes(x_unlabeled)

    n_candidates = min(candidate_size, n_unlabeled)
    cand_idx = floor_stratified_sample(
        codes,
        n_samples=n_candidates,
        floor_per_bucket=floor_per_bucket,
        seed=seed,
    )
    x_cand = x_unlabeled[cand_idx]

    maha = MahalanobisGate(quantile=maha_quantile).fit(x_cand)
    knn = KNNDensityGate(quantile=knn_quantile).fit(x_cand)

    vote_maha = maha.vote(x_cand)
    vote_knn = knn.vote(x_cand)
    votes = vote_maha.to(torch.int64) + vote_knn.to(torch.int64)

    harvested_mask = votes >= vote_threshold
    harvested_idx_all = cand_idx[harvested_mask]

    harvested_idx = harvested_idx_all
    if target_harvest_size is not None and harvested_idx_all.numel() > target_harvest_size:
        codes_h = codes[harvested_idx_all]
        picked_rel = floor_stratified_sample(
            codes_h,
            n_samples=target_harvest_size,
            floor_per_bucket=post_floor_per_bucket,
            seed=seed + 1,
        )
        harvested_idx = harvested_idx_all[picked_rel]

    stats = HarvestStats(
        n_unlabeled=n_unlabeled,
        n_candidates=n_candidates,
        n_harvested=int(harvested_idx.numel()),
        pass_maha=int(vote_maha.sum().item()),
        pass_knn=int(vote_knn.sum().item()),
    )
    return harvested_idx, stats
