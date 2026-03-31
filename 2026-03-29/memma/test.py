from __future__ import annotations

import os
from pathlib import Path

from dataset import make_dataset
from agent import MemoryManager, MetaThinker, QueryReasoner, SelfEvolver


def main() -> None:
    data = make_dataset(n=10, seed=42)
    ep = data[0]

    meta = MetaThinker(D=48)
    manager = MemoryManager()
    reasoner = QueryReasoner()
    evolver = SelfEvolver(probe_steps=3)

    store_idx = meta.select_to_store(ep.keys, ep.values, ep.key_emb, budget=4)
    manager.build(ep.keys, ep.values, ep.key_emb, store_idx)

    evolver.evolve(manager=manager, keys=ep.keys, values=ep.values, emb=ep.key_emb)
    retrieved = manager.retrieve(ep.query_emb, topk=5)
    pred, _ = reasoner.answer(retrieved, ep.query_key)

    assert isinstance(pred, str)
    print("memma smoke test ok")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
