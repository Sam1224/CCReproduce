from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .toy_dataset import Sample


@dataclass(frozen=True)
class PairSample:
    """A training sample after ES^3-style expansion.

    For simplicity, we train a single binary label (click) and optionally a second head (purchase).
    """

    user_id: int
    query_id: int
    item_id: int
    domain_id: int
    user_tokens: np.ndarray
    query_tokens: np.ndarray
    item_tokens: np.ndarray
    click: int
    purchase: int
    exposed: int  # 1 if originally exposed, 0 if unexposed negative


def sample_unexposed_negatives(
    *,
    rng: np.random.Generator,
    query_id: int,
    positive_item_id: int,
    item_pool: Sequence[int],
    query_vectors: Dict[int, np.ndarray],
    item_vectors: Dict[int, np.ndarray],
    k: int,
) -> List[int]:
    """Sample K hard-ish negatives from the item pool.

    This approximates "entire-space" candidate expansion by sampling items that are
    somewhat similar to the query but not the positive item.
    """

    qv = query_vectors[query_id]
    pool = np.array(item_pool, dtype=np.int64)

    # Get a small candidate set, then pick top-similarity ones
    cand = rng.choice(pool, size=min(len(pool), max(200, 10 * k)), replace=False)
    sims = np.array([float(qv @ item_vectors[int(i)]) for i in cand], dtype=np.float32)

    order = np.argsort(-sims)
    negatives: List[int] = []
    for idx in order:
        item_id = int(cand[idx])
        if item_id == positive_item_id:
            continue
        negatives.append(item_id)
        if len(negatives) >= k:
            break

    if len(negatives) < k:
        # fallback random
        while len(negatives) < k:
            item_id = int(rng.choice(pool))
            if item_id != positive_item_id:
                negatives.append(item_id)

    return negatives


def es3_expand(
    *,
    seed: int,
    logs: Iterable[Sample],
    item_vectors: Dict[int, np.ndarray],
    query_vectors: Dict[int, np.ndarray],
    neg_per_pos: int = 20,
    cross_domain_ratio: float = 0.2,
) -> List[PairSample]:
    """Toy ES^3 expansion.

    - For each clicked sample: add unexposed negatives.
    - For cross-domain: convert a subset of samples into pseudo-search samples by
      scrambling query tokens ("searchification").

    Returns a flattened list used for training.
    """

    rng = np.random.default_rng(seed)
    item_pool = list(item_vectors.keys())

    expanded: List[PairSample] = []
    logs_list = list(logs)

    for s in logs_list:
        expanded.append(
            PairSample(
                user_id=s.user_id,
                query_id=s.query_id,
                item_id=s.item_id,
                domain_id=s.domain_id,
                user_tokens=s.user_tokens,
                query_tokens=s.query_tokens,
                item_tokens=s.item_tokens,
                click=s.click,
                purchase=s.purchase,
                exposed=1,
            )
        )

        if s.click:
            negs = sample_unexposed_negatives(
                rng=rng,
                query_id=s.query_id,
                positive_item_id=s.item_id,
                item_pool=item_pool,
                query_vectors=query_vectors,
                item_vectors=item_vectors,
                k=neg_per_pos,
            )
            for neg_item_id in negs:
                expanded.append(
                    PairSample(
                        user_id=s.user_id,
                        query_id=s.query_id,
                        item_id=neg_item_id,
                        domain_id=s.domain_id,
                        user_tokens=s.user_tokens,
                        query_tokens=s.query_tokens,
                        item_tokens=s.item_tokens,  # keep original item tokens; toy simplification
                        click=0,
                        purchase=0,
                        exposed=0,
                    )
                )

    # cross-domain searchification (toy): convert some samples into new domain with scrambled queries
    m = int(len(expanded) * cross_domain_ratio)
    for _ in range(m):
        base = expanded[int(rng.integers(0, len(expanded)))]
        q = base.query_tokens.copy()
        rng.shuffle(q)
        new_domain = int(rng.integers(0, 3))
        expanded.append(
            PairSample(
                user_id=base.user_id,
                query_id=base.query_id,
                item_id=base.item_id,
                domain_id=new_domain,
                user_tokens=base.user_tokens,
                query_tokens=q,
                item_tokens=base.item_tokens,
                click=base.click,
                purchase=base.purchase,
                exposed=base.exposed,
            )
        )

    return expanded
