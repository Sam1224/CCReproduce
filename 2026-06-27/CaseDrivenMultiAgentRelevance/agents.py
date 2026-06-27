from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import random

import torch

from data import ToyRelevanceDataset, ToyRelevanceConfig, encode_tokens
from eval import evaluate
from train import train


@dataclass
class Standards:
    """Toy stand-in for natural-language standards.

    We model standards as explicit rules + a synonym table used by the annotator
    and deep-search expansion.
    """

    synonyms: Dict[str, Tuple[str, ...]]
    must_match_color: bool = True
    require_wireless_if_asked: bool = True


@dataclass
class GlobalMemory:
    resolved_cases: List[Tuple[str, str, int]] = field(default_factory=list)
    standard_updates: List[str] = field(default_factory=list)

    def add_case(self, q: str, d: str, y: int) -> None:
        self.resolved_cases.append((q, d, y))

    def add_update(self, text: str) -> None:
        self.standard_updates.append(text)


class UserAgent:
    def mine_bad_cases(self, model, ds: ToyRelevanceDataset, top_k: int = 200):
        # select mispredicted cases, biased toward underestimation (pred < gold)
        model.eval()
        rows = []
        for ex in ds.items:
            out = model(ex["q_ids"].unsqueeze(0), ex["d_ids"].unsqueeze(0))
            pred = int(out["coarse_logits"].argmax(dim=-1).item())
            gold = int(ex["y"].item())
            if pred != gold:
                rows.append((gold - pred, ex, pred, gold))

        rows.sort(key=lambda t: (t[0], t[3]), reverse=True)
        return rows[:top_k]


class DeepSearchAgent:
    def expand_query(self, q: str, standards: Standards) -> List[str]:
        toks = q.split()
        expanded = set(toks)
        for t in toks:
            for s in standards.synonyms.get(t, ()): 
                expanded.add(s)
        return list(expanded)


class AnnotatorAgent:
    def __init__(self, cfg: ToyRelevanceConfig):
        self.cfg = cfg

    def label(self, q_text: str, d_text: str, standards: Standards) -> int:
        # multi-path reasoning: vote among three heuristics
        votes = [
            self._label_rule_based(q_text, d_text, standards),
            self._label_overlap(q_text, d_text),
            self._label_with_synonyms(q_text, d_text, standards),
        ]
        return int(sorted(votes)[len(votes) // 2])

    def _label_rule_based(self, q_text: str, d_text: str, standards: Standards) -> int:
        q = q_text.split()
        d = d_text.split()

        # crude category inference
        q_cat = q[-1]
        d_cat = d[-1]
        if q_cat != d_cat:
            return 0

        if standards.must_match_color:
            colors = {"red", "blue", "black", "white", "green"}
            q_color = next((t for t in q if t in colors), None)
            d_color = next((t for t in d if t in colors), None)
            if q_color and d_color and q_color != d_color:
                return 1

        if standards.require_wireless_if_asked and "wireless" in q and "wireless" not in d:
            return 1

        return 3 if len(set(q) & set(d)) >= 2 else 2

    def _label_overlap(self, q_text: str, d_text: str) -> int:
        q = q_text.split()
        d = d_text.split()
        if q[-1] != d[-1]:
            return 0
        overlap = len(set(q) & set(d))
        if overlap >= 3:
            return 3
        if overlap == 2:
            return 2
        return 1

    def _label_with_synonyms(self, q_text: str, d_text: str, standards: Standards) -> int:
        q = q_text.split()
        d = d_text.split()
        if q[-1] != d[-1]:
            return 0

        exp = set(q)
        for t in q:
            exp.update(standards.synonyms.get(t, ()))
        overlap = len(exp & set(d))
        return 3 if overlap >= 3 else 2 if overlap == 2 else 1


class OptimizerAgent:
    def __init__(self, cfg: ToyRelevanceConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def optimize(self, model, train_ds: ToyRelevanceDataset, dev_ds: ToyRelevanceDataset, bad_cases, standards: Standards, memory: GlobalMemory):
        # 1) relabel a slice of bad cases
        annotator = AnnotatorAgent(self.cfg)
        augmented = []
        for _, ex, _, _ in bad_cases:
            q = ex["q_text"]
            d = ex["d_text"]
            y_new = annotator.label(q, d, standards)
            memory.add_case(q, d, y_new)

            augmented.append((q, d, y_new))
            # 2) synonym augmentation (DeepSearch-style)
            if self.rng.random() < 0.6:
                for t, syns in list(standards.synonyms.items())[:3]:
                    if t in q:
                        q2 = q.replace(t, self.rng.choice(syns))
                        augmented.append((q2, d, y_new))

        # 3) merge augmented samples into training set
        for q, d, y in augmented:
            train_ds.items.append(
                {
                    "q_text": q,
                    "d_text": d,
                    "q_ids": encode_tokens(q.split(), self.cfg),
                    "d_ids": encode_tokens(d.split(), self.cfg),
                    "y": torch.tensor(int(y), dtype=torch.long),
                }
            )

        before = evaluate(model, dev_ds)
        train(model, train_ds, epochs=2, batch_size=128, lr=2e-3)
        after = evaluate(model, dev_ds)

        # record a tiny "standard update" when the win-rate improves
        if after.win_rate > before.win_rate + 1e-6:
            memory.add_update(f"win_rate {before.win_rate:.3f} -> {after.win_rate:.3f}")

        return before, after
