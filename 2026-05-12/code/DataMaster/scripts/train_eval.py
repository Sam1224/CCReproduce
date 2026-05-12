from __future__ import annotations

import argparse
import os

from datamaster.core.state import DataState
from datamaster.models.text_classifier import TrainConfig, train_and_eval
from datamaster.tasks.toy_content_moderation.dataset import build_toy_task


def _texts_labels(records):
    texts = [r["text"] for r in records]
    labels = [int(r["label"]) for r in records]
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="baseline", choices=["baseline", "base+external_clean", "base+external_noisy"])
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    splits = build_toy_task(data_dir)

    if args.recipe == "baseline":
        train_records = splits.pool.get("base")
    elif args.recipe == "base+external_clean":
        train_records = splits.pool.get("base") + splits.pool.get("external_clean")
    else:
        train_records = splits.pool.get("base") + splits.pool.get("external_noisy")

    val_records = splits.val_records

    train_texts, train_labels = _texts_labels(train_records)
    val_texts, val_labels = _texts_labels(val_records)

    metrics = train_and_eval(train_texts, train_labels, val_texts, val_labels, cfg=TrainConfig(epochs=6))
    print(f"recipe={args.recipe}")
    print(metrics)


if __name__ == "__main__":
    main()
