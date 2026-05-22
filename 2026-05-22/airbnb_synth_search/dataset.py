import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_WORD_RE = re.compile(r"[a-z0-9']+")


def load_seed_queries(path: str) -> List[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_listings(path: str) -> Dict[str, dict]:
    listings: Dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            listings[obj["listing_id"]] = obj
    return listings


def load_sessions(path: str) -> List[dict]:
    sessions: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def build_listing_text(listing: dict) -> str:
    parts: List[str] = []
    parts.append(listing.get("property_type", "place"))
    if listing.get("location"):
        parts.append(f"in {listing['location']}")
    if listing.get("city"):
        parts.append(listing["city"])
    if listing.get("price_tier"):
        parts.append(f"price {listing['price_tier']}")
    for a in listing.get("amenities", []):
        parts.append(a)
    for v in listing.get("vibe", []):
        parts.append(v)
    for n in listing.get("nearby", []):
        parts.append(f"near {n}")
    return " ".join(parts)


@dataclass
class Vocab:
    token_to_id: Dict[str, int]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "Vocab":
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for tok, c in sorted(freq.items()):
            if c >= min_freq and tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(t, 1) for t in tokenize(text)]


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class RuleBasedLLMStub:
    """A deterministic stand-in for LLM prompting.

    In the real system, an LLM is prompted with: seed query + contrastive listing pair.
    Here we craft a plausible query that (a) stays close to the seed style, and
    (b) uses salient attributes from the positive listing.
    """

    def generate(self, prompt: dict) -> str:
        seed = prompt["seed_query"].strip()
        hints = prompt["hints"]
        city = hints.get("city")
        prop = hints.get("property_type")
        amenity = hints.get("amenity")
        vibe = hints.get("vibe")

        seed_core = seed
        if city and city.lower() not in seed_core.lower():
            seed_core = f"{seed_core} in {city}"

        extras = [x for x in [prop, amenity, vibe] if x]
        extra = ", ".join(extras[:2]) if extras else ""

        if extra:
            return f"{seed_core}, ideally a {extra}".strip()
        return seed_core

    def virtual_judge_score(self, query: str, listing: dict) -> float:
        """Heuristic 'virtual judge' score, higher means more relevant."""
        q = set(tokenize(query))
        l = set(tokenize(build_listing_text(listing)))
        if not q:
            return 0.0
        return len(q & l) / len(q)


class TriplesDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Vocab):
        self.rows: List[dict] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        r = self.rows[idx]
        return (
            self.vocab.encode(r["query"]),
            self.vocab.encode(r["pos_text"]),
            self.vocab.encode(r["neg_text"]),
        )


def collate_batch(batch: List[Tuple[List[int], List[int], List[int]]]):
    def pack(seqs: List[List[int]]):
        offsets = [0]
        flat: List[int] = []
        for s in seqs:
            flat.extend(s)
            offsets.append(len(flat))
        return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets[:-1], dtype=torch.long)

    q, p, n = zip(*batch)
    q_flat, q_off = pack(list(q))
    p_flat, p_off = pack(list(p))
    n_flat, n_off = pack(list(n))
    return (q_flat, q_off), (p_flat, p_off), (n_flat, n_off)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def length_histogram(texts: List[str], max_len: int = 40) -> np.ndarray:
    hist = np.zeros(max_len + 1, dtype=np.float64)
    for t in texts:
        l = min(len(tokenize(t)), max_len)
        hist[l] += 1
    return hist
