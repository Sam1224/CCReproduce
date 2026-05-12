from __future__ import annotations

import os

from datamaster.core.datatree import DataMasterSearch, SearchConfig
from datamaster.core.state import DataState
from datamaster.models.text_classifier import TrainConfig, train_and_eval
from datamaster.tasks.toy_content_moderation.dataset import build_toy_task


def _texts_labels(records):
    texts = [r["text"] for r in records]
    labels = [int(r["label"]) for r in records]
    return texts, labels


def main() -> None:
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    artifacts_dir = os.path.join(root, "artifacts")

    splits = build_toy_task(data_dir)
    pool = splits.pool
    val_records = splits.val_records
    val_texts, val_labels = _texts_labels(val_records)

    def evaluate(state: DataState):
        train_texts, train_labels = _texts_labels(state.records)
        metrics = train_and_eval(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            cfg=TrainConfig(epochs=5, batch_size=32, lr=1e-2),
        )
        score = float(metrics["f1"])
        return score, metrics

    initial_state = DataState(records=pool.get("base"), sources=["base"], ops=[])

    search = DataMasterSearch(
        pool=pool,
        evaluate=evaluate,
        config=SearchConfig(max_depth=3, beam_size=6, max_expansions=20, artifacts_dir=artifacts_dir),
    )

    best = search.run(initial_state)

    print("=== Best Node ===")
    print(f"node_id={best.node_id} depth={best.depth} score={best.score:.4f}")
    print(f"sources={best.state.sources}")
    print(f"ops={best.state.ops}")
    print(best.metrics)


if __name__ == "__main__":
    main()
