from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ToyRetrievalExample:
    query_ids: List[int]
    pos_doc_id: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rand_seq(rng: random.Random, vocab_size: int, length: int, avoid: set[int] | None = None) -> List[int]:
    avoid = avoid or set()
    out: List[int] = []
    while len(out) < length:
        token_id = rng.randint(1, vocab_size - 1)
        if token_id in avoid:
            continue
        out.append(token_id)
    return out


def generate_toy_dataset(
    out_dir: str,
    *,
    vocab_size: int = 2000,
    num_docs: int = 2000,
    doc_len: int = 48,
    num_queries: int = 500,
    query_len: int = 16,
    overlap: int = 10,
    seed: int = 7,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)

    docs: List[List[int]] = []
    for _ in range(num_docs):
        docs.append(_rand_seq(rng, vocab_size, doc_len))

    queries: List[ToyRetrievalExample] = []
    for _ in range(num_queries):
        pos_doc_id = rng.randint(0, num_docs - 1)
        doc = docs[pos_doc_id]

        shared = rng.sample(doc, k=min(overlap, len(doc)))
        avoid = set(shared)
        noise = _rand_seq(rng, vocab_size, max(0, query_len - len(shared)), avoid=avoid)
        q = shared + noise
        rng.shuffle(q)
        queries.append(ToyRetrievalExample(query_ids=q, pos_doc_id=pos_doc_id))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "num_docs": num_docs,
                "doc_len": doc_len,
                "num_queries": num_queries,
                "query_len": query_len,
                "overlap": overlap,
                "seed": seed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(out_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, token_ids in enumerate(docs):
            f.write(json.dumps({"doc_id": doc_id, "token_ids": token_ids}) + "\n")

    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid, ex in enumerate(queries):
            f.write(
                json.dumps({"qid": qid, "query_ids": ex.query_ids, "pos_doc_id": ex.pos_doc_id})
                + "\n"
            )


def load_docs(data_dir: str) -> List[List[int]]:
    docs: List[List[int]] = []
    with open(os.path.join(data_dir, "docs.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(row["token_ids"])
    return docs


class ToyQueryDataset(Dataset):
    def __init__(self, data_dir: str):
        self.examples: List[ToyRetrievalExample] = []
        with open(os.path.join(data_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.examples.append(ToyRetrievalExample(query_ids=row["query_ids"], pos_doc_id=row["pos_doc_id"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ToyRetrievalExample:
        return self.examples[idx]


def collate_queries(batch: List[ToyRetrievalExample]) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex.query_ids) for ex in batch)
    query_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    query_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    pos_doc_ids = torch.tensor([ex.pos_doc_id for ex in batch], dtype=torch.long)

    for i, ex in enumerate(batch):
        query_ids[i, : len(ex.query_ids)] = torch.tensor(ex.query_ids, dtype=torch.long)
        query_mask[i, : len(ex.query_ids)] = True

    return {"query_ids": query_ids, "query_mask": query_mask, "pos_doc_ids": pos_doc_ids}
