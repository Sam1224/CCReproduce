import argparse
import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    doc_id: int
    query: str
    answer: str
    paragraphs: list[str]
    gold_pid: int


def make_document(rng: np.random.Generator, doc_id: int, num_paragraphs: int, noise_ratio: float):
    entity = f"entity_{int(rng.integers(0, 50))}"
    attr = f"attr_{int(rng.integers(0, 10))}"
    gold_value = f"value_{int(rng.integers(0, 200))}"

    paragraphs = []
    gold_pid = int(rng.integers(0, num_paragraphs))

    for pid in range(num_paragraphs):
        if pid == gold_pid:
            paragraphs.append(f"Fact: {entity} {attr} is {gold_value}.")
            continue

        if rng.random() < noise_ratio:
            if rng.random() < 0.5:
                wrong_attr = f"attr_{int(rng.integers(0, 10))}"
                wrong_value = f"value_{int(rng.integers(0, 200))}"
                paragraphs.append(f"Rumor: {entity} {wrong_attr} is {wrong_value}.")
            else:
                wrong_value = f"value_{int(rng.integers(0, 200))}"
                paragraphs.append(f"Rumor: {entity} {attr} is {wrong_value}.")
        else:
            other_entity = f"entity_{int(rng.integers(0, 50))}"
            other_attr = f"attr_{int(rng.integers(0, 10))}"
            other_value = f"value_{int(rng.integers(0, 200))}"
            paragraphs.append(f"Fact: {other_entity} {other_attr} is {other_value}.")

    query = f"Question: What is {entity} {attr}?"
    return Sample(doc_id=doc_id, query=query, answer=gold_value, paragraphs=paragraphs, gold_pid=gold_pid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_docs", type=int, default=400)
    parser.add_argument("--num_paragraphs", type=int, default=24)
    parser.add_argument("--noise_ratio", type=float, default=0.55)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    samples = [
        make_document(rng, doc_id=i, num_paragraphs=args.num_paragraphs, noise_ratio=args.noise_ratio)
        for i in range(args.num_docs)
    ]

    rng.shuffle(samples)
    split = int(args.num_docs * args.train_ratio)
    train, test = samples[:split], samples[split:]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for s in items:
                f.write(
                    json.dumps(
                        {
                            "doc_id": s.doc_id,
                            "query": s.query,
                            "answer": s.answer,
                            "paragraphs": s.paragraphs,
                            "gold_pid": s.gold_pid,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    dump(os.path.join(args.out_dir, "train.jsonl"), train)
    dump(os.path.join(args.out_dir, "test.jsonl"), test)

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "num_docs": args.num_docs,
                "num_paragraphs": args.num_paragraphs,
                "noise_ratio": args.noise_ratio,
                "train_ratio": args.train_ratio,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
