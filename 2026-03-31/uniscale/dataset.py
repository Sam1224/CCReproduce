from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RequestCandidate:
    request_id: int
    domain_id: int
    user_dense: torch.Tensor
    ctx_dense: torch.Tensor
    item_dense: torch.Tensor
    user_interest_dense: torch.Tensor
    clicked: float
    purchased: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ES3Builder:
    """ES^3-style data constructor.

    This is a *reproduction-oriented* implementation that keeps the main ideas:

    - Intra-domain sample expansion: add unexposed candidates to mitigate selection bias.
    - Hierarchical label attribution: propagate delayed conversions to earlier interactions.
    - Cross-domain searchification: map non-search behaviors into a search-like feature schema.

    The real paper uses billion-scale logs + production feature systems; here we provide:

    - A synthetic generator (default)
    - A hook for loading real request logs into the same interface.
    """

    def __init__(
        self,
        *,
        d_user: int = 32,
        d_ctx: int = 16,
        d_item: int = 32,
        d_interest: int = 16,
        n_domains: int = 6,
        candidate_pool: int = 2000,
        seed: int = 7,
    ) -> None:
        self.d_user = d_user
        self.d_ctx = d_ctx
        self.d_item = d_item
        self.d_interest = d_interest
        self.n_domains = n_domains
        self.candidate_pool = candidate_pool
        self.seed = seed

        seed_everything(seed)

        self._domain_emb = torch.randn(n_domains, d_interest)
        self._item_bank = torch.randn(candidate_pool, d_item)

    def _user_profile(self) -> torch.Tensor:
        return torch.randn(self.d_user)

    def _ctx_profile(self) -> torch.Tensor:
        return torch.randn(self.d_ctx)

    def _sample_items(self, k: int) -> torch.Tensor:
        idx = torch.randint(0, self.candidate_pool, (k,))
        return self._item_bank[idx]

    def _cross_domain_interest(self, user: torch.Tensor, domain_id: int) -> torch.Tensor:
        base = self._domain_emb[domain_id]
        proj = torch.randn(self.d_user, self.d_interest)
        return torch.tanh(user @ proj + base)

    def _label_from_score(self, s: float) -> Tuple[float, float]:
        click_p = _sigmoid(s)
        purchase_p = _sigmoid(s - 1.0)
        click = float(np.random.rand() < click_p)
        purchase = float(click and (np.random.rand() < purchase_p))
        return click, purchase

    def build_requests(
        self,
        *,
        n_requests: int,
        candidates: int,
        exposed_ratio: float = 0.3,
        unexposed_ratio: float = 0.7,
        cross_domain_ratio: float = 0.25,
    ) -> List[RequestCandidate]:
        """Return *candidate-level* samples across requests.

        Each request yields `candidates` samples with (clicked, purchased) labels.
        """

        all_samples: List[RequestCandidate] = []
        for rid in range(n_requests):
            domain_id = int(np.random.randint(0, self.n_domains))
            user = self._user_profile()
            ctx = self._ctx_profile()

            # Candidate set.
            item_dense = self._sample_items(candidates)

            # Exposed vs unexposed.
            n_exposed = max(1, int(candidates * exposed_ratio))
            exposed_mask = torch.zeros(candidates, dtype=torch.bool)
            exposed_mask[:n_exposed] = True
            exposed_mask = exposed_mask[torch.randperm(candidates)]

            # Cross-domain events (simulated) for hierarchical attribution.
            cross_domain_purchase = float(np.random.rand() < cross_domain_ratio)

            interest = self._cross_domain_interest(user, domain_id)

            # Score function mimics ranking relevance + user interest.
            w_user = torch.randn(self.d_user)
            w_ctx = torch.randn(self.d_ctx)
            w_item = torch.randn(self.d_item)
            w_interest = torch.randn(self.d_interest)

            for j in range(candidates):
                s = (
                    float(user @ w_user)
                    + float(ctx @ w_ctx)
                    + float(item_dense[j] @ w_item)
                    + float(interest @ w_interest)
                    + float(self._domain_emb[domain_id] @ w_interest) * 0.5
                )

                click, purchase = self._label_from_score(s)

                # (1) Intra-domain sample expansion: unexposed items become easy negatives
                # unless they later receive attributed conversion.
                if not bool(exposed_mask[j]):
                    click = 0.0
                    purchase = 0.0

                # (2) Hierarchical label attribution: propagate delayed conversion signals.
                # We simulate: if any cross-domain purchase happens, attribute partial
                # credit to earlier clicked items in this request.
                if cross_domain_purchase and click > 0:
                    purchase = max(purchase, 0.5)  # soft label for attributed conversion

                # (3) Cross-domain searchification: for non-search domains, map behaviors
                # into aligned features (here: affine perturbation).
                if domain_id != 0:
                    align = torch.randn(self.d_item, self.d_item) / math.sqrt(self.d_item)
                    item_feat = torch.tanh(item_dense[j] @ align)
                else:
                    item_feat = item_dense[j]

                all_samples.append(
                    RequestCandidate(
                        request_id=rid,
                        domain_id=domain_id,
                        user_dense=user,
                        ctx_dense=ctx,
                        item_dense=item_feat,
                        user_interest_dense=interest,
                        clicked=float(click),
                        purchased=float(purchase),
                    )
                )

        # Apply unexposed_ratio by additional negative sampling (search-space uniform).
        if unexposed_ratio > 0:
            extra = int(len(all_samples) * unexposed_ratio)
            for k in range(extra):
                rid = int(np.random.randint(0, n_requests))
                domain_id = int(np.random.randint(0, self.n_domains))
                user = self._user_profile()
                ctx = self._ctx_profile()
                item = self._sample_items(1).squeeze(0)
                interest = self._cross_domain_interest(user, domain_id)
                all_samples.append(
                    RequestCandidate(
                        request_id=rid,
                        domain_id=domain_id,
                        user_dense=user,
                        ctx_dense=ctx,
                        item_dense=item,
                        user_interest_dense=interest,
                        clicked=0.0,
                        purchased=0.0,
                    )
                )

        return all_samples


class UniScaleDataset(Dataset):
    def __init__(self, samples: List[RequestCandidate]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        y = torch.tensor([s.purchased if s.purchased > 0 else s.clicked], dtype=torch.float32)
        return {
            "request_id": torch.tensor([s.request_id], dtype=torch.long),
            "domain_id": torch.tensor([s.domain_id], dtype=torch.long),
            "user_dense": s.user_dense.float(),
            "ctx_dense": s.ctx_dense.float(),
            "item_dense": s.item_dense.float(),
            "user_interest_dense": s.user_interest_dense.float(),
            "label": y,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        if k in {"user_dense", "ctx_dense", "item_dense", "user_interest_dense"}:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


def iter_request_groups(request_ids: torch.Tensor) -> Iterable[torch.Tensor]:
    """Yield index tensors grouping a batch by request_id."""
    ids = request_ids.detach().cpu().tolist()
    groups: Dict[int, List[int]] = {}
    for i, rid in enumerate(ids):
        groups.setdefault(int(rid), []).append(i)
    for _, idxs in groups.items():
        if len(idxs) >= 2:
            yield torch.tensor(idxs, dtype=torch.long)
