from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ToyConfig:
    n_items: int = 6000
    n_queries_train: int = 24000
    n_queries_test: int = 6000
    query_min_attrs: int = 2
    query_max_attrs: int = 4

    codebooks: Tuple[str, ...] = (
        "category",
        "brand",
        "color",
        "style",
        "price",
        "material",
    )

    reserved_per_codebook: int = 2

    # editability toy setting
    injected_keyword: str = "trend:viral"
    injected_codebook: str = "style"
    injected_res_idx: int = 0
    injected_target_frac: float = 0.06


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int
    unk_id: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _choices(prefix: str, n: int) -> List[str]:
    return [f"{prefix}:{i}" for i in range(n)]


def build_attribute_pools() -> Dict[str, List[str]]:
    return {
        "category": _choices("cat", 30),
        "brand": _choices("brand", 80),
        "color": _choices("color", 18),
        "style": _choices("style", 50),
        "price": _choices("price", 12),
        "material": _choices("mat", 22),
    }


def build_items(cfg: ToyConfig, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    pools = build_attribute_pools()

    items = []
    for i in range(cfg.n_items):
        attrs = {cb: rng.choice(pools[cb]) for cb in cfg.codebooks}
        # item code tokens are exactly the aligned attribute tokens
        code = [attrs[cb] for cb in cfg.codebooks]
        items.append({"item_id": i, "attrs": attrs, "code": code})

    return items


def build_codebook_vocabs(cfg: ToyConfig, items: List[Dict]) -> Dict[str, List[str]]:
    vocabs: Dict[str, List[str]] = {cb: [] for cb in cfg.codebooks}
    for cb in cfg.codebooks:
        uniq = sorted({it["attrs"][cb] for it in items})
        reserved = [f"RES_{cb.upper()}_{j}" for j in range(cfg.reserved_per_codebook)]
        vocabs[cb] = uniq + reserved
    return vocabs


def build_query_vocab(cfg: ToyConfig, items: List[Dict]) -> List[str]:
    base = sorted({t for it in items for t in it["code"]})
    # injected keyword exists in vocab but is NOT used in training queries
    fillers = ["find", "need", "cheap", "good", "for", "gift", "new"]
    return base + fillers + [cfg.injected_keyword]


def build_vocab(cfg: ToyConfig, items: List[Dict]) -> Tuple[Vocab, Dict[str, Dict[str, int]]]:
    token_to_id: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    id_to_token = ["[PAD]", "[UNK]"]

    for t in build_query_vocab(cfg, items):
        if t not in token_to_id:
            token_to_id[t] = len(id_to_token)
            id_to_token.append(t)

    slot_maps: Dict[str, Dict[str, int]] = {}
    for cb, tokens in build_codebook_vocabs(cfg, items).items():
        slot_maps[cb] = {t: i for i, t in enumerate(tokens)}

    vocab = Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)
    return vocab, slot_maps


def encode_query(vocab: Vocab, tokens: List[str], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = [vocab.token_to_id.get(t, vocab.unk_id) for t in tokens][:max_len]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        pad = [vocab.pad_id] * (max_len - len(ids))
        ids = ids + pad
        mask = mask + [0] * len(pad)
    return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.int64)


def _make_query_tokens(
    rng: random.Random, cfg: ToyConfig, item: Dict, allow_injected: bool
) -> List[str]:
    attrs = item["code"]
    n = rng.randint(cfg.query_min_attrs, cfg.query_max_attrs)
    chosen = rng.sample(attrs, k=n)

    tokens = ["find", "need"] + chosen

    # occasionally add a random filler
    if rng.random() < 0.3:
        tokens.append(rng.choice(["cheap", "good", "gift", "new"]))

    if allow_injected and rng.random() < 0.35:
        tokens.append(cfg.injected_keyword)

    rng.shuffle(tokens)
    return tokens


def build_queries(
    cfg: ToyConfig,
    items: List[Dict],
    vocab: Vocab,
    slot_maps: Dict[str, Dict[str, int]],
    seed: int,
    max_len: int = 16,
) -> Dict[str, np.ndarray]:
    rng = random.Random(seed)

    def code_ids(item: Dict) -> np.ndarray:
        arr = np.zeros(len(cfg.codebooks), dtype=np.int64)
        for i, cb in enumerate(cfg.codebooks):
            arr[i] = slot_maps[cb][item["attrs"][cb]]
        return arr

    xs_train, ms_train, ys_train = [], [], []
    for _ in range(cfg.n_queries_train):
        it = items[rng.randrange(len(items))]
        tokens = _make_query_tokens(rng, cfg, it, allow_injected=False)
        x, m = encode_query(vocab, tokens, max_len=max_len)
        xs_train.append(x)
        ms_train.append(m)
        ys_train.append(code_ids(it))

    xs_test, ms_test, ys_test = [], [], []
    for _ in range(cfg.n_queries_test):
        it = items[rng.randrange(len(items))]
        tokens = _make_query_tokens(rng, cfg, it, allow_injected=False)
        x, m = encode_query(vocab, tokens, max_len=max_len)
        xs_test.append(x)
        ms_test.append(m)
        ys_test.append(code_ids(it))

    return {
        "x_train": np.stack(xs_train),
        "m_train": np.stack(ms_train),
        "y_train": np.stack(ys_train),
        "x_test": np.stack(xs_test),
        "m_test": np.stack(ms_test),
        "y_test": np.stack(ys_test),
    }


def build_editability_suite(
    cfg: ToyConfig,
    items: List[Dict],
    vocab: Vocab,
    slot_maps: Dict[str, Dict[str, int]],
    seed: int,
    max_len: int = 16,
) -> Dict:
    rng = random.Random(seed + 11)

    # pick a target set to be "injected" by a new keyword
    n_target = max(10, int(math.floor(cfg.n_items * cfg.injected_target_frac)))
    target_ids = rng.sample(range(cfg.n_items), k=n_target)

    injected_cb_idx = cfg.codebooks.index(cfg.injected_codebook)
    injected_token = f"RES_{cfg.injected_codebook.upper()}_{cfg.injected_res_idx}"
    injected_token_id = slot_maps[cfg.injected_codebook][injected_token]

    # patched codes for index-time intervention
    item_code_ids = np.zeros((cfg.n_items, len(cfg.codebooks)), dtype=np.int64)
    patched_code_ids = np.zeros_like(item_code_ids)

    for it in items:
        i = it["item_id"]
        for j, cb in enumerate(cfg.codebooks):
            tid = slot_maps[cb][it["attrs"][cb]]
            item_code_ids[i, j] = tid
            patched_code_ids[i, j] = tid

    for i in target_ids:
        patched_code_ids[i, injected_cb_idx] = injected_token_id

    # construct injected queries: common attributes + injected keyword
    injected_queries = []
    for _ in range(400):
        base_item = items[rng.choice(target_ids)]
        tokens = _make_query_tokens(rng, cfg, base_item, allow_injected=True)
        if cfg.injected_keyword not in tokens:
            tokens.append(cfg.injected_keyword)
        x, m = encode_query(vocab, tokens, max_len=max_len)
        injected_queries.append((x, m))

    return {
        "target_ids": np.asarray(sorted(target_ids), dtype=np.int64),
        "injected_cb_idx": injected_cb_idx,
        "injected_token_id": int(injected_token_id),
        "item_code_ids": item_code_ids,
        "patched_code_ids": patched_code_ids,
        "injected_queries": injected_queries,
        "injected_keyword": cfg.injected_keyword,
    }


def hamming_rank(code: np.ndarray, codes: np.ndarray, topk: int) -> np.ndarray:
    # code: [C], codes: [N, C]
    dist = (codes != code[None, :]).sum(axis=1)
    idx = np.argsort(dist, kind="stable")
    return idx[:topk]


def hit_rate_at_k(pred_rank: np.ndarray, true_id: int) -> float:
    return float(true_id in set(pred_rank.tolist()))


def evaluate_hr(
    pred_codes: np.ndarray,
    true_codes: np.ndarray,
    item_code_ids: np.ndarray,
    topk: int,
) -> float:
    # for each query, treat its true positive as the item whose code matches y
    # (if multiple items share the same code, pick the first match)
    hits = []
    for pc, tc in zip(pred_codes, true_codes):
        # find a deterministic positive item id
        matches = np.where((item_code_ids == tc[None, :]).all(axis=1))[0]
        true_id = int(matches[0]) if len(matches) else 0
        rank = hamming_rank(pc, item_code_ids, topk=topk)
        hits.append(hit_rate_at_k(rank, true_id))
    return float(np.mean(hits))


def evaluate_injection(
    pred_codes: np.ndarray,
    patched_code_ids: np.ndarray,
    target_ids: np.ndarray,
    topk: int,
) -> Tuple[float, float]:
    iars = []
    ihrs = []
    target_set = set(target_ids.tolist())

    for pc in pred_codes:
        rank = hamming_rank(pc, patched_code_ids, topk=topk)
        hit_targets = [i for i in rank.tolist() if i in target_set]
        ihrs.append(float(len(hit_targets) > 0))
        iars.append(float(len(hit_targets) / min(topk, len(target_set))))

    return float(np.mean(iars)), float(np.mean(ihrs))
