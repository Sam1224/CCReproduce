from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Session:
    sid: str
    evidence_tokens: List[str]
    target_tokens: List[str]


_BRANDS = ["acme", "freshco", "greenfarm", "snacklab", "vita", "yummy"]
_CATS = ["coffee", "tea", "snacks", "cereal", "sauce", "noodles"]
_FLAVORS = ["vanilla", "chocolate", "spicy", "sweet", "original", "lemon"]
_DIET = ["regular", "gluten_free", "lactose_free", "vegan"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


SPECIALS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<vid>",
    "</vid>",
    "<hist>",
    "</hist>",
    "<meta>",
    "</meta>",
]


def build_vocab(samples: Sequence[Session]) -> Dict[str, int]:
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for s in samples:
        for t in s.evidence_tokens + s.target_tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get("<pad>", 0)
    return [vocab.get(t, unk) for t in tokens]


def compress(tokens: List[str], max_len: int) -> List[str]:
    """Prompt compression proxy.

    OneBar uses prompt compression for latency. Here we keep a fixed budget by
    preserving segment markers and truncating inside segments.
    """

    if len(tokens) <= max_len:
        return tokens

    keep = []
    for t in tokens:
        if t in {"<vid>", "</vid>", "<meta>", "</meta>", "<hist>", "</hist>", "<sep>", "<bos>", "<eos>"}:
            keep.append(t)

    # Ensure markers exist.
    if "<vid>" not in keep:
        keep = ["<vid>", "</vid>"] + keep

    # Reserve space for markers; allocate remaining budget to content tokens.
    budget = max(0, max_len - len(keep))

    content = [t for t in tokens if t not in set(keep)]
    return keep + content[:budget]


def make_sessions(n: int, seed: int = 0, max_prompt_len: int = 36) -> List[Session]:
    set_seed(seed)
    out: List[Session] = []

    for i in range(n):
        brand = random.choice(_BRANDS)
        cat = random.choice(_CATS)
        flavor = random.choice(_FLAVORS)
        diet = random.choice(_DIET)

        # user history (noisy, sometimes conflicting)
        hist = []
        for _ in range(random.randint(0, 3)):
            hb = random.choice(_BRANDS)
            hc = random.choice(_CATS)
            hist.extend([hb, hc, "<sep>"])

        meta = [f"brand:{brand}", f"cat:{cat}", f"diet:{diet}"]
        vid = [f"video:{brand}", f"video:{cat}", f"video:{flavor}"]

        evidence = ["<bos>", "<vid>"] + vid + ["</vid>", "<meta>"] + meta + ["</meta>"]
        if hist:
            evidence += ["<hist>"] + hist + ["</hist>"]

        evidence = compress(evidence, max_prompt_len)

        # target query (what the model should generate)
        tgt = [brand, cat]
        if random.random() < 0.35:
            tgt.append(flavor)
        if random.random() < 0.25:
            tgt.append(diet)

        out.append(Session(sid=f"s{i:05d}", evidence_tokens=evidence, target_tokens=tgt))

    return out


def split(samples: Sequence[Session], seed: int = 0, ratios=(0.8, 0.1, 0.1)):
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    n = len(samples)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    tr = [samples[i] for i in idx[:n_tr]]
    va = [samples[i] for i in idx[n_tr : n_tr + n_va]]
    te = [samples[i] for i in idx[n_tr + n_va :]]
    return tr, va, te
