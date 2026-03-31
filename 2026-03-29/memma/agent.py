from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class MemoryItem:
    key: str
    value: str
    emb: torch.Tensor  # (D,)


class MetaThinker:
    """Produces structured guidance for what to store.

    In the real paper this would be LLM reasoning. Here we simulate a policy that
    stores a subset of facts (and can be tuned by a learnable gate).
    """

    def __init__(self, *, D: int = 48):
        self.scorer = torch.nn.Linear(D, 1)

    def select_to_store(self, keys: List[str], values: List[str], emb: torch.Tensor, *, budget: int) -> List[int]:
        # (N,)
        s = self.scorer(emb).squeeze(-1)
        k = min(budget, emb.shape[0])
        idx = torch.topk(s, k=k, dim=0).indices.tolist()
        return idx


class MemoryManager:
    def __init__(self):
        self.items: List[MemoryItem] = []

    def build(self, keys: List[str], values: List[str], emb: torch.Tensor, store_idx: List[int]) -> None:
        self.items = [MemoryItem(key=keys[i], value=values[i], emb=emb[i].detach().clone()) for i in store_idx]

    def retrieve(self, query_emb: torch.Tensor, *, topk: int = 5) -> List[MemoryItem]:
        if not self.items:
            return []
        E = torch.stack([it.emb for it in self.items], dim=0)  # (M, D)
        q = query_emb.view(1, -1)
        sim = torch.nn.functional.cosine_similarity(E, q, dim=-1)
        k = min(topk, len(self.items))
        idx = torch.topk(sim, k=k).indices.tolist()
        return [self.items[i] for i in idx]

    def repair_add(self, key: str, value: str, emb: torch.Tensor) -> None:
        # avoid duplicates
        if any(it.key == key for it in self.items):
            return
        self.items.append(MemoryItem(key=key, value=value, emb=emb.detach().clone()))


class QueryReasoner:
    def answer(self, retrieved: List[MemoryItem], query_key: str) -> Tuple[str, bool]:
        for it in retrieved:
            if it.key == query_key:
                return it.value, True
        return "UNKNOWN", False


class SelfEvolver:
    """Backward path: probe, verify, repair before finalizing memory."""

    def __init__(self, *, probe_steps: int = 4):
        self.probe_steps = probe_steps

    def evolve(self, *, manager: MemoryManager, keys: List[str], values: List[str], emb: torch.Tensor) -> int:
        repaired = 0
        # probe random keys (in real MemMA: synthesize probe QA pairs)
        for i in range(min(self.probe_steps, len(keys))):
            k_idx = int(torch.randint(0, len(keys), (1,)).item())
            probe_key = keys[k_idx]
            retrieved = manager.retrieve(emb[k_idx], topk=3)
            ok = any(it.key == probe_key for it in retrieved)
            if not ok:
                manager.repair_add(probe_key, values[k_idx], emb[k_idx])
                repaired += 1
        return repaired
