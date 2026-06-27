from __future__ import annotations

import copy
import random

from agents import DeepSearchAgent, GlobalMemory, OptimizerAgent, Standards, UserAgent
from data import ToyRelevanceConfig, ToyRelevanceDataset, default_synonyms
from eval import evaluate
from model import AllInOneConfig, AllInOneRelevanceModel
from train import train


def main() -> None:
    cfg = ToyRelevanceConfig(synonyms=default_synonyms())

    train_ds = ToyRelevanceDataset(8000, cfg, seed=0)
    dev_ds = ToyRelevanceDataset(1500, cfg, seed=1)

    model = AllInOneRelevanceModel(AllInOneConfig(vocab_size=cfg.vocab_size, max_len=cfg.max_len))

    print("== baseline training ==")
    train(model, train_ds, epochs=1)
    base = evaluate(model, dev_ds)
    print(f"baseline acc={base.acc:.3f} macro_f1={base.macro_f1:.3f} win_rate={base.win_rate:.3f}")

    standards = Standards(synonyms=cfg.synonyms)
    memory = GlobalMemory()

    user = UserAgent()
    deep = DeepSearchAgent()
    opt = OptimizerAgent(cfg, rng=random.Random(42))

    print("\n== multi-agent iterations ==")
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
        print(
            f"iter={it} win_rate={before.win_rate:.3f}->{after.win_rate:.3f} "
            f"(gain={gain*100:.2f}pp) acc={after.acc:.3f} f1={after.macro_f1:.3f}"
        )

        if after.win_rate > best.win_rate + 1e-6:
            best = after
            best_state = copy.deepcopy(model.state_dict())
        else:
            print("early-stop: no win_rate improvement")
            break

    model.load_state_dict(best_state)
    print("\n== best snapshot ==")
    print(f"best acc={best.acc:.3f} macro_f1={best.macro_f1:.3f} win_rate={best.win_rate:.3f}")

    print("\n== memory snapshot ==")
    print(f"resolved_cases={len(memory.resolved_cases)} standard_updates={len(memory.standard_updates)}")
    for u in memory.standard_updates[:8]:
        print(f"- {u}")


if __name__ == "__main__":
    main()
