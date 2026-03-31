from __future__ import annotations

import os
from pathlib import Path

import torch

from agent import MemoryManager, MetaThinker, QueryReasoner, SelfEvolver
from dataset import make_dataset, split


def run(eval_data, *, budget: int, evolve: bool) -> float:
    meta = MetaThinker(D=48)
    reasoner = QueryReasoner()
    evolver = SelfEvolver(probe_steps=6)

    correct = 0
    total = 0
    for ep in eval_data:
        manager = MemoryManager()
        store_idx = meta.select_to_store(ep.keys, ep.values, ep.key_emb, budget=budget)
        manager.build(ep.keys, ep.values, ep.key_emb, store_idx)

        if evolve:
            evolver.evolve(manager=manager, keys=ep.keys, values=ep.values, emb=ep.key_emb)

        retrieved = manager.retrieve(ep.query_emb, topk=5)
        pred, ok = reasoner.answer(retrieved, ep.query_key)
        correct += int(ok and pred == ep.answer)
        total += 1

    return correct / max(1, total)


def main() -> None:
    data = make_dataset(n=1500, seed=1)
    _, te = split(data, 0.85)

    acc_no = run(te[:600], budget=8, evolve=False)
    acc_yes = run(te[:600], budget=8, evolve=True)

    print(f"acc(no-evolve)={acc_no:.3f}")
    print(f"acc(with-evolve)={acc_yes:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
