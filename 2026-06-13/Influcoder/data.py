from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

from model import ToyConfig, Vocab


TOXIC_WORDS = [
    "hate",
    "abuse",
    "toxic",
    "harass",
    "slur",
]

TOPICS = [
    "shopping",
    "fashion",
    "makeup",
    "electronics",
    "home",
    "fitness",
    "travel",
    "food",
    "books",
]

TEMPLATES = [
    "user says: {topic} {sent} please",
    "assistant reply about {topic}: {sent}",
    "query: {topic} {sent}",
]

SAFE_SENT = [
    "great quality",
    "good price",
    "recommend it",
    "looks nice",
    "fast delivery",
    "highly rated",
    "durable and light",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_vocab(cfg: ToyConfig, texts: List[str]) -> Vocab:
    # simple whitespace vocab
    freq: Dict[str, int] = {}
    for t in texts:
        for tok in t.split():
            freq[tok] = freq.get(tok, 0) + 1

    # reserve pad/unk
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    id_to_token = ["<pad>", "<unk>"] + [w for w, _ in items[: max(0, cfg.vocab_size - 2)]]
    token_to_id = {t: i for i, t in enumerate(id_to_token)}

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def encode(vocab: Vocab, text: str, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    toks = text.split()[:max_len]
    ids = np.full((max_len,), vocab.pad_id, dtype=np.int64)
    mask = np.zeros((max_len,), dtype=np.int64)

    for i, t in enumerate(toks):
        ids[i] = vocab.token_to_id.get(t, vocab.unk_id)
        mask[i] = 1

    return ids, mask


def build_toy_dataset(cfg: ToyConfig, seed: int = 7) -> Dict:
    set_seed(seed)

    pool_texts: List[str] = []
    pool_y: List[int] = []

    for _ in range(cfg.n_pool):
        topic = random.choice(TOPICS)
        if random.random() < cfg.toxic_rate:
            w = random.choice(TOXIC_WORDS)
            sent = f"{w} content"
            y = 1
        else:
            sent = random.choice(SAFE_SENT)
            y = 0
        txt = random.choice(TEMPLATES).format(topic=topic, sent=sent)
        pool_texts.append(txt)
        pool_y.append(y)

    # queries ask for "find toxic source" like in data attribution
    query_texts: List[str] = []
    for _ in range(cfg.n_query):
        topic = random.choice(TOPICS)
        w = random.choice(TOXIC_WORDS)
        query_texts.append(f"analyze {topic} {w} source")

    all_texts = pool_texts + query_texts
    vocab = build_vocab(cfg, all_texts)

    pool_x = []
    pool_m = []
    for t in pool_texts:
        x, m = encode(vocab, t, cfg.max_len)
        pool_x.append(x)
        pool_m.append(m)

    query_x = []
    query_m = []
    for t in query_texts:
        x, m = encode(vocab, t, cfg.max_len)
        query_x.append(x)
        query_m.append(m)

    return {
        "cfg": asdict(cfg),
        "vocab": {"id_to_token": vocab.id_to_token, "pad_id": vocab.pad_id, "unk_id": vocab.unk_id},
        "pool": {
            "texts": pool_texts,
            "x": np.stack(pool_x),
            "m": np.stack(pool_m),
            "y": np.asarray(pool_y, dtype=np.int64),
        },
        "query": {
            "texts": query_texts,
            "x": np.stack(query_x),
            "m": np.stack(query_m),
        },
    }
