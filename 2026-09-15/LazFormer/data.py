"""Synthetic data for LazFormer (toy, runnable).

We build two domains:

- Source domain (generative pre-training):
  request-conditioned sequences. The generator mostly follows an item-level
  transition table (a within-category "next-item" ring), so autoregressive
  pretraining is meaningful.

- Target domain (ranking fine-tuning):
  (request, long history, candidate set) with:
  (1) a domain shift in request->category mapping, and
  (2) a shifted/noised transition table.

The positive item is defined as:

  pos = transition_target[ last_item_in_history_matching_request ]

This makes ranking genuinely *request-aware* and *history-dependent* (mean pooling
baselines struggle when multiple interests co-exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PAD = 0  # padding item id; real items are 1..num_items


@dataclass
class DataConfig:
    # shared
    num_items: int = 800
    num_categories: int = 20
    num_requests: int = 16
    seed: int = 0

    # pretrain
    pretrain_num_seqs: int = 5000
    pretrain_min_len: int = 12
    pretrain_max_len: int = 35
    p_follow_transition: float = 0.85

    # rank
    rank_train: int = 3500
    rank_valid: int = 600
    rank_test: int = 600
    history_len: int = 50
    num_candidates: int = 60

    # user behavior
    p_switch_interest: float = 0.15

    # request-aware compression
    topm_history: int = 12


def _build_item_bank(num_items: int, num_categories: int, rng: np.random.Generator):
    item_cat = rng.integers(0, num_categories, size=num_items, dtype=np.int64)
    cat_items: Dict[int, np.ndarray] = {}
    for c in range(num_categories):
        idx = np.where(item_cat == c)[0] + 1  # item ids are 1-indexed
        if len(idx) == 0:
            idx = np.array([rng.integers(1, num_items + 1)], dtype=np.int64)
        cat_items[c] = idx
    return item_cat, cat_items


def _build_request_profiles(
    num_requests: int,
    num_categories: int,
    rng: np.random.Generator,
    *,
    domain: str,
):
    """Return request -> preferred categories.

    domain='source': stable mapping
    domain='target': permuted mapping + random extra categories (domain shift)
    """
    if domain not in {"source", "target"}:
        raise ValueError(f"unknown domain={domain}")

    base = np.arange(num_categories)
    if domain == "target":
        rng.shuffle(base)

    req_prefs: List[List[int]] = []
    for r in range(num_requests):
        c0 = int(base[r % num_categories])
        if domain == "source":
            c1 = int(base[(r + 3) % num_categories])
            prefs = [c0, c1]
        else:
            extra = list(map(int, rng.choice(num_categories, size=2, replace=False)))
            prefs = [c0] + extra
        req_prefs.append(prefs)
    return req_prefs


def _build_transition_table(
    num_items: int,
    item_cat: np.ndarray,
    cat_items: Dict[int, np.ndarray],
    rng: np.random.Generator,
    *,
    domain: str,
):
    """Item->next-item table.

    - source: within-category ring shift by 1
    - target: per-category random shift + noise (10% items remapped within-category)
    """
    if domain not in {"source", "target"}:
        raise ValueError(f"unknown domain={domain}")

    trans = np.zeros(num_items + 1, dtype=np.int64)

    for c, items in cat_items.items():
        if len(items) == 1:
            trans[items[0]] = items[0]
            continue
        if domain == "source":
            shift = 1
        else:
            shift = int(rng.integers(1, min(5, len(items))))
        trans[items] = np.roll(items, -shift)

    if domain == "target":
        n_noise = max(1, num_items // 10)
        for _ in range(n_noise):
            i = int(rng.integers(1, num_items + 1))
            c = int(item_cat[i - 1])
            trans[i] = int(rng.choice(cat_items[c]))

    return trans


def _sample_sequence_by_transition(
    rng: np.random.Generator,
    *,
    length: int,
    trans: np.ndarray,
    cat_items: Dict[int, np.ndarray],
    req_prefs: List[List[int]],
    req_id: int,
    p_follow_transition: float,
    p_switch_interest: float,
):
    prefs = req_prefs[req_id]
    cur_cat = int(rng.choice(prefs))
    cur_item = int(rng.choice(cat_items[cur_cat]))

    seq: List[int] = []
    for _ in range(length):
        if rng.random() < p_follow_transition:
            nxt = int(trans[cur_item])
        else:
            if rng.random() < p_switch_interest:
                cur_cat = int(rng.choice(prefs))
            nxt = int(rng.choice(cat_items[cur_cat]))
        seq.append(cur_item)
        cur_item = nxt
    return seq


class PretrainSeqDataset(Dataset):
    """(request, item_sequence) for autoregressive next-item pretraining."""

    def __init__(self, cfg: DataConfig, *, domain: str = "source") -> None:
        super().__init__()
        rng = np.random.default_rng(cfg.seed + (11 if domain == "source" else 17))
        self.cfg = cfg
        self.domain = domain

        self.item_cat, self.cat_items = _build_item_bank(cfg.num_items, cfg.num_categories, rng)
        self.req_prefs = _build_request_profiles(cfg.num_requests, cfg.num_categories, rng, domain=domain)
        self.trans = _build_transition_table(cfg.num_items, self.item_cat, self.cat_items, rng, domain=domain)

        self.samples: List[Tuple[int, List[int]]] = []
        for _ in range(cfg.pretrain_num_seqs):
            req = int(rng.integers(0, cfg.num_requests))
            length = int(rng.integers(cfg.pretrain_min_len, cfg.pretrain_max_len + 1))
            seq = _sample_sequence_by_transition(
                rng,
                length=length,
                trans=self.trans,
                cat_items=self.cat_items,
                req_prefs=self.req_prefs,
                req_id=req,
                p_follow_transition=cfg.p_follow_transition,
                p_switch_interest=cfg.p_switch_interest,
            )
            self.samples.append((req, seq))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class RankDataset(Dataset):
    """(request, history, candidates, label_index) for target-domain ranking."""

    def __init__(self, cfg: DataConfig, *, split: str, domain: str = "target") -> None:
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"unknown split={split}")

        # fixed item bank & request profiles across splits
        bank_rng = np.random.default_rng(cfg.seed + 201)
        prof_rng = np.random.default_rng(cfg.seed + (202 if domain == "target" else 203))
        trans_rng = np.random.default_rng(cfg.seed + (204 if domain == "target" else 205))

        self.cfg = cfg
        self.split = split
        self.domain = domain

        self.item_cat, self.cat_items = _build_item_bank(cfg.num_items, cfg.num_categories, bank_rng)
        self.req_prefs = _build_request_profiles(cfg.num_requests, cfg.num_categories, prof_rng, domain=domain)
        self.trans = _build_transition_table(cfg.num_items, self.item_cat, self.cat_items, trans_rng, domain=domain)

        split_seed = {"train": 101, "valid": 102, "test": 103}[split]
        rng = np.random.default_rng(cfg.seed + split_seed)

        n = {"train": cfg.rank_train, "valid": cfg.rank_valid, "test": cfg.rank_test}[split]
        self.samples: List[Tuple[int, List[int], List[int], int]] = []

        for _ in range(n):
            req = int(rng.integers(0, cfg.num_requests))
            primary_cat = int(self.req_prefs[req][0])
            # distractor category makes the task order-sensitive: the request anchor is NOT the last item
            other_cat = int(rng.choice([c for c in range(cfg.num_categories) if c != primary_cat]))

            L1 = cfg.history_len // 2
            L2 = cfg.history_len // 3
            L3 = cfg.history_len - L1 - L2

            def sample_segment(cat: int, length: int) -> List[int]:
                item = int(rng.choice(self.cat_items[cat]))
                seg: List[int] = []
                for _ in range(length):
                    seg.append(item)
                    item = int(self.trans[item])
                return seg

            # segment-1: primary intent
            seg1 = sample_segment(primary_cat, L1)
            # segment-2: distractor (placed after seg1 so seg1's last is the request anchor)
            seg2 = sample_segment(other_cat, L2)
            # segment-3: more distractors (never primary)
            seg3 = sample_segment(other_cat, L3) if L3 > 0 else []

            hist = seg1 + seg2 + seg3
            anchor = seg1[-1]
            pos = int(self.trans[anchor])

            # build candidates: pos + hard negatives
            cand: List[int] = [pos]

            # hard negatives: next items of other recent anchors (confusable)
            recent = hist[-10:]
            rng.shuffle(recent)
            for x in recent[: min(10, len(recent))]:
                neg = int(self.trans[x])
                if neg != pos:
                    cand.append(neg)

            # in-category negatives (hard): make category-level matching insufficient
            in_cat_items = self.cat_items[primary_cat]
            seen = set(cand)
            while len(cand) < cfg.num_candidates:
                if rng.random() < 0.85:
                    neg = int(rng.choice(in_cat_items))
                else:
                    neg = int(rng.integers(1, cfg.num_items + 1))
                if neg in seen:
                    continue
                seen.add(neg)
                cand.append(neg)

            cand = cand[: cfg.num_candidates]

            perm = rng.permutation(len(cand))
            cand = [cand[i] for i in perm]
            label = int(np.where(perm == 0)[0][0])

            self.samples.append((req, hist, cand, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_pretrain(batch):
    req = torch.tensor([b[0] for b in batch], dtype=torch.long)
    seqs = [b[1] for b in batch]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    L = int(lengths.max().item())
    out = torch.zeros(len(batch), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return {"req": req, "seq": out, "lengths": lengths}


def collate_rank(batch):
    req = torch.tensor([b[0] for b in batch], dtype=torch.long)
    hist = torch.tensor([b[1] for b in batch], dtype=torch.long)
    cands = torch.tensor([b[2] for b in batch], dtype=torch.long)
    label = torch.tensor([b[3] for b in batch], dtype=torch.long)
    lengths = torch.full((len(batch),), hist.shape[1], dtype=torch.long)
    return {"req": req, "hist": hist, "lengths": lengths, "cands": cands, "label": label}


@torch.no_grad()
def recall_ndcg(scores: torch.Tensor, label: torch.Tensor, ks=(5, 10)):
    """scores: [B,K], label: [B] index of positive."""
    B, K = scores.shape
    order = scores.argsort(dim=1, descending=True)
    pos = label.view(B, 1)
    match = (order == pos).nonzero(as_tuple=False)
    rank_idx = torch.full((B,), K - 1, dtype=torch.long, device=scores.device)
    rank_idx[match[:, 0]] = match[:, 1]
    rank1 = rank_idx + 1

    out = {}
    for k in ks:
        hit = (rank1 <= k).float()
        out[f"Recall@{k}"] = float(hit.mean().item())
        denom = torch.log2(rank1.float() + 1.0)
        dcg = torch.where(rank1 <= k, 1.0 / denom, torch.zeros_like(denom))
        out[f"NDCG@{k}"] = float(dcg.mean().item())
    return out


def build_cfg_from_ckpt(ckpt: dict) -> DataConfig:
    return DataConfig(**ckpt["data_cfg"])


if __name__ == "__main__":
    cfg = DataConfig(seed=0)
    ds = PretrainSeqDataset(cfg, domain="source")
    print("pretrain", len(ds), "example_len", len(ds[0][1]))
    rk = RankDataset(cfg, split="train", domain="target")
    print("rank", len(rk), "cand", len(rk[0][2]), "label", rk[0][3])
