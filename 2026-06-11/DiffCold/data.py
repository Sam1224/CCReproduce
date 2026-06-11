import math
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class ToyRecConfig:
    num_users: int = 2000
    num_warm_items: int = 1200
    num_cold_items: int = 300
    content_dim: int = 64
    embed_dim: int = 64
    warm_pos_per_user: int = 40
    cold_pos_per_user: int = 10
    train_pos_per_user: int = 30
    k_retrieve: int = 10


def _nonlinear_behavior_embedding(content: np.ndarray, embed_dim: int, rng: np.random.Generator) -> np.ndarray:
    w1 = rng.normal(scale=1.0 / math.sqrt(content.shape[1]), size=(content.shape[1], embed_dim))
    w2 = rng.normal(scale=1.0 / math.sqrt(embed_dim), size=(embed_dim, embed_dim))

    h = np.tanh(content @ w1)
    h = np.tanh(h @ w2)
    # add a light manifold warp
    h = h + 0.05 * np.sin(h)
    h = h + rng.normal(scale=0.05, size=h.shape)
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    return h.astype(np.float32)


def generate_toy_data(cfg: ToyRecConfig, seed: int = 7):
    rng = np.random.default_rng(seed)

    num_items = cfg.num_warm_items + cfg.num_cold_items

    content = rng.normal(size=(num_items, cfg.content_dim)).astype(np.float32)
    content = content / (np.linalg.norm(content, axis=1, keepdims=True) + 1e-12)

    item_emb_true = _nonlinear_behavior_embedding(content, cfg.embed_dim, rng)

    user_emb = rng.normal(size=(cfg.num_users, cfg.embed_dim)).astype(np.float32)
    user_emb = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-12)

    warm_ids = np.arange(cfg.num_warm_items)
    cold_ids = np.arange(cfg.num_warm_items, num_items)

    warm_pos = []
    cold_pos = []
    for u in range(cfg.num_users):
        scores_warm = user_emb[u] @ item_emb_true[warm_ids].T
        # encourage diversity
        probs = np.exp(scores_warm / 0.2)
        probs = probs / probs.sum()
        pos_w = rng.choice(warm_ids, size=cfg.warm_pos_per_user, replace=False, p=probs)

        scores_cold = user_emb[u] @ item_emb_true[cold_ids].T
        probs_c = np.exp(scores_cold / 0.2)
        probs_c = probs_c / probs_c.sum()
        pos_c = rng.choice(cold_ids, size=cfg.cold_pos_per_user, replace=False, p=probs_c)

        warm_pos.append(pos_w)
        cold_pos.append(pos_c)

    warm_pos = np.stack(warm_pos, axis=0)
    cold_pos = np.stack(cold_pos, axis=0)

    train_warm = warm_pos[:, : cfg.train_pos_per_user]
    test_warm = warm_pos[:, cfg.train_pos_per_user :]

    return {
        "content": content,
        "item_emb_true": item_emb_true,
        "user_emb": user_emb,
        "warm_ids": warm_ids,
        "cold_ids": cold_ids,
        "train_warm": train_warm,
        "test_warm": test_warm,
        "test_cold": cold_pos,
    }


def build_retrieval_aggregator(content: np.ndarray, warm_ids: np.ndarray, warm_item_emb: np.ndarray, k: int = 10):
    warm_content = content[warm_ids]

    # cosine similarity since content already normalized
    sim = content @ warm_content.T

    # top-k warm neighbors for each item
    topk = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]

    agg = np.zeros((content.shape[0], warm_item_emb.shape[1]), dtype=np.float32)
    for i in range(content.shape[0]):
        nbr_idx = topk[i]
        agg[i] = warm_item_emb[warm_ids[nbr_idx]].mean(axis=0)

    agg = agg / (np.linalg.norm(agg, axis=1, keepdims=True) + 1e-12)
    return agg


def recall_ndcg_at_k(rank_lists: np.ndarray, positives: np.ndarray, k: int = 20):
    # rank_lists: [U, N] of item ids sorted by score desc
    # positives:  [U, P]
    u = rank_lists.shape[0]
    recall_sum = 0.0
    ndcg_sum = 0.0

    for i in range(u):
        pos = set(int(x) for x in positives[i])
        topk = rank_lists[i, :k]
        hits = [1 if int(x) in pos else 0 for x in topk]

        num_pos = len(pos)
        recall_sum += sum(hits) / max(1, num_pos)

        dcg = 0.0
        for j, h in enumerate(hits, start=1):
            if h:
                dcg += 1.0 / math.log2(j + 1)

        ideal_hits = [1] * min(num_pos, k)
        idcg = 0.0
        for j, h in enumerate(ideal_hits, start=1):
            idcg += 1.0 / math.log2(j + 1)

        ndcg_sum += dcg / max(1e-12, idcg)

    return recall_sum / u, ndcg_sum / u
