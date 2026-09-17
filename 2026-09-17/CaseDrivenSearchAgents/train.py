from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DataConfig, ToyRelevanceDataset, build_cfg_from_ckpt, default_synonyms, encode_tokens, evaluate
from model import AllInOneRelevanceModel, ModelConfig

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
BASELINE_CKPT = CKPT_DIR / "baseline.pt"
AGENT_CKPT = CKPT_DIR / "multi_agent.pt"


@dataclass
class Standards:
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
    def mine_bad_cases(self, model, ds: ToyRelevanceDataset, top_k: int = 120):
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
            expanded.update(standards.synonyms.get(t, ()))
        return list(expanded)


class AnnotatorAgent:
    def label(self, q_text: str, d_text: str, standards: Standards) -> int:
        votes = [
            self._label_rule_based(q_text, d_text, standards),
            self._label_overlap(q_text, d_text),
            self._label_with_synonyms(q_text, d_text, standards),
        ]
        return int(sorted(votes)[len(votes) // 2])

    def _label_rule_based(self, q_text: str, d_text: str, standards: Standards) -> int:
        q = q_text.split()
        d = d_text.split()
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
    def __init__(self, cfg: DataConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def optimize(self, model, train_ds: ToyRelevanceDataset, dev_ds: ToyRelevanceDataset, bad_cases, standards: Standards, memory: GlobalMemory):
        annotator = AnnotatorAgent()
        augmented = []
        for _, ex, _, _ in bad_cases:
            q = ex["q_text"]
            d = ex["d_text"]
            y_new = annotator.label(q, d, standards)
            memory.add_case(q, d, y_new)
            augmented.append((q, d, y_new))
            if self.rng.random() < 0.6:
                for t, syns in list(standards.synonyms.items())[:3]:
                    if t in q:
                        q2 = q.replace(t, self.rng.choice(syns))
                        augmented.append((q2, d, y_new))
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
        train_loop(model, train_ds, epochs=2, batch_size=128, lr=2e-3)
        after = evaluate(model, dev_ds)
        if after.win_rate > before.win_rate + 1e-6:
            memory.add_update(f"win_rate {before.win_rate:.3f} -> {after.win_rate:.3f}")
        return before, after


def train_loop(model, train_ds, epochs: int = 5, batch_size: int = 128, lr: float = 2e-3) -> None:
    model.train()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in loader:
            out = model(batch["q_ids"], batch["d_ids"])
            y = batch["y"]
            y_retrieval = (y > 0).to(torch.float32)
            loss_coarse = F.cross_entropy(out["coarse_logits"], y)
            loss_retrieval = F.binary_cross_entropy_with_logits(out["retrieval_logit"], y_retrieval)
            y_fine = (y.to(torch.float32) / 3.0).clamp(0, 1)
            loss_fine = F.mse_loss(torch.sigmoid(out["fine_score"]), y_fine)
            loss = loss_coarse + 0.3 * loss_retrieval + 0.2 * loss_fine
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch} loss={total/len(loader):.4f}")


def save_checkpoint(path: Path, model, cfg: DataConfig, mcfg: ModelConfig, tag: str, memory: GlobalMemory | None = None) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tag": tag,
            "data_cfg": {**cfg.__dict__, "synonyms": cfg.synonyms or default_synonyms()},
            "model_cfg": mcfg.__dict__,
            "state_dict": model.state_dict(),
            "memory": {
                "resolved_cases": memory.resolved_cases if memory else [],
                "standard_updates": memory.standard_updates if memory else [],
            },
        },
        path,
    )
    print(f"saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = DataConfig(synonyms=default_synonyms())
    train_ds = ToyRelevanceDataset(8000, cfg, seed=args.seed)
    dev_ds = ToyRelevanceDataset(1500, cfg, seed=args.seed + 1)
    mcfg = ModelConfig(vocab_size=cfg.vocab_size, max_len=cfg.max_len)

    baseline = AllInOneRelevanceModel(mcfg)
    print("== baseline training ==")
    train_loop(baseline, train_ds, epochs=1)
    base = evaluate(baseline, dev_ds)
    print("baseline:", base)
    save_checkpoint(BASELINE_CKPT, baseline, cfg, mcfg, "baseline")

    model = AllInOneRelevanceModel(mcfg)
    model.load_state_dict(copy.deepcopy(baseline.state_dict()))
    standards = Standards(synonyms=cfg.synonyms)
    memory = GlobalMemory()
    user = UserAgent()
    deep = DeepSearchAgent()
    opt = OptimizerAgent(cfg, rng=random.Random(42))

    print("== multi-agent iterations ==")
    best = base
    best_state = copy.deepcopy(model.state_dict())
    for it in range(1, 4):
        bad = user.mine_bad_cases(model, dev_ds, top_k=120)
        if bad:
            q0 = bad[0][1]["q_text"]
            expanded = deep.expand_query(q0, standards)
            if len(expanded) > len(q0.split()):
                memory.add_update(f"DeepSearch expanded query: {q0} -> {' '.join(expanded)}")
        before, after = opt.optimize(model, train_ds, dev_ds, bad, standards, memory)
        gain = after.win_rate - before.win_rate
        print(f"iter={it} win_rate={before.win_rate:.3f}->{after.win_rate:.3f} gain={gain*100:.2f}pp acc={after.acc:.3f} f1={after.macro_f1:.3f}")
        if after.win_rate > best.win_rate + 1e-6:
            best = after
            best_state = copy.deepcopy(model.state_dict())
        else:
            break
    model.load_state_dict(best_state)
    print("best:", best)
    save_checkpoint(AGENT_CKPT, model, cfg, mcfg, "multi_agent", memory)


if __name__ == "__main__":
    main()
