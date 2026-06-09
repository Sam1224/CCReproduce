from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ToyRecBatch:
    history_item_ids: torch.LongTensor  # [B, H]
    target_item_id: torch.LongTensor  # [B]
    target_semantic_ids: torch.LongTensor  # [B, L]


@dataclass
class ToyRecWorld:
    # Catalog definition
    item_emb: torch.FloatTensor  # [N, D]
    semantic_ids: torch.LongTensor  # [N, L]
    seq_to_item: dict[tuple[int, ...], int]

    # Popularity + exposure bias simulation
    popularity: torch.FloatTensor  # [N]
    rm_bias: torch.FloatTensor  # [N]


def build_world(
    *,
    num_items: int = 200,
    token_vocab: int = 64,
    semantic_len: int = 3,
    emb_dim: int = 32,
    seed: int = 0,
) -> ToyRecWorld:
    rng = np.random.default_rng(seed)

    item_emb = torch.tensor(rng.normal(size=(num_items, emb_dim)).astype(np.float32))
    item_emb = torch.nn.functional.normalize(item_emb, dim=-1)

    # Build unique semantic ID sequences
    semantic_ids = []
    seen: set[tuple[int, ...]] = set()
    while len(semantic_ids) < num_items:
        seq = tuple(int(x) for x in rng.integers(0, token_vocab, size=(semantic_len,)).tolist())
        if seq in seen:
            continue
        seen.add(seq)
        semantic_ids.append(seq)

    semantic_ids_t = torch.tensor(semantic_ids, dtype=torch.long)
    seq_to_item = {tuple(seq): i for i, seq in enumerate(semantic_ids)}

    # Simulate popularity and exposure bias: head items get positive bias
    raw_pop = rng.pareto(1.2, size=(num_items,)).astype(np.float32) + 1e-3
    popularity = torch.tensor(raw_pop / raw_pop.sum(), dtype=torch.float32)

    # Reward model bias correlated with popularity (exposure bias)
    rm_bias = torch.log(popularity * num_items + 1e-6)
    rm_bias = (rm_bias - rm_bias.mean()) / (rm_bias.std() + 1e-6)
    rm_bias = 0.15 * rm_bias  # control bias strength

    return ToyRecWorld(
        item_emb=item_emb,
        semantic_ids=semantic_ids_t,
        seq_to_item=seq_to_item,
        popularity=popularity,
        rm_bias=rm_bias,
    )


def _sample_user_pref(rng: np.random.Generator, emb_dim: int) -> np.ndarray:
    v = rng.normal(size=(emb_dim,)).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-6
    return v


def sample_batch(
    world: ToyRecWorld,
    *,
    batch_size: int,
    history_len: int,
    seed: int,
) -> ToyRecBatch:
    rng = np.random.default_rng(seed)

    num_items, emb_dim = world.item_emb.shape
    item_emb = world.item_emb

    histories = []
    target_items = []

    for _ in range(batch_size):
        user_pref = torch.tensor(_sample_user_pref(rng, emb_dim))
        scores = (item_emb @ user_pref).cpu().numpy()
        target = int(scores.argmax())

        # build history: mostly from top items, plus noise
        topk = np.argsort(-scores)[:50]
        noise = rng.choice(num_items, size=(history_len,), replace=True)
        hist = []
        for i in range(history_len):
            if rng.random() < 0.8:
                hist.append(int(rng.choice(topk)))
            else:
                hist.append(int(noise[i]))
        histories.append(hist)
        target_items.append(target)

    history_item_ids = torch.tensor(histories, dtype=torch.long)
    target_item_id = torch.tensor(target_items, dtype=torch.long)
    target_semantic_ids = world.semantic_ids[target_item_id]

    return ToyRecBatch(
        history_item_ids=history_item_ids,
        target_item_id=target_item_id,
        target_semantic_ids=target_semantic_ids,
    )
