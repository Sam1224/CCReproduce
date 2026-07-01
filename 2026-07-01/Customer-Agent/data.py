from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

PRODUCT_TYPES = ["BOOK", "HEADBAND", "DVD", "CAMERA", "LAPTOP", "TOY", "STEREO", "COFFEE"]
BRANDS = ["Aster", "North", "Roewell", "Zen", "Pixel", "Nova"]


@dataclass
class QAExample:
    trajectory_id: int
    question: str
    sql_template: int
    answer_value: int


def synthesize_trajectory(path: Path, tid: int, rows: int = 80, seed: int = 0) -> None:
    rng = random.Random(seed + tid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "action_type", "product_types", "brand", "price"])
        writer.writeheader()
        for i in range(rows):
            ptype = rng.choice(PRODUCT_TYPES)
            writer.writerow({
                "row_id": i,
                "action_type": rng.choice(["search", "click", "purchase", "click"]),
                "product_types": ptype,
                "brand": rng.choice(BRANDS),
                "price": round(rng.uniform(5, 900), 2),
            })


def build_toy_files(root: Path, n: int = 96) -> None:
    for tid in range(n):
        synthesize_trajectory(root / f"trajectory_{tid}.csv", tid, seed=7)


class ShopTrajQADataset(Dataset):
    def __init__(self, root: Path, n: int = 96):
        self.root = root
        build_toy_files(root, n=n)
        self.examples: list[QAExample] = []
        for tid in range(n):
            path = root / f"trajectory_{tid}.csv"
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            purchases = [r for r in rows if r["action_type"] == "purchase"]
            clicks = [r for r in rows if r["action_type"] == "click"]
            self.examples.append(QAExample(tid, "How many purchases are in the trajectory?", 0, len(purchases)))
            self.examples.append(QAExample(tid, "How many clicked items are expensive, price over 500?", 1, sum(float(r["price"]) > 500 for r in clicks)))
            most = max(PRODUCT_TYPES, key=lambda t: sum(r["product_types"] == t for r in rows))
            self.examples.append(QAExample(tid, f"How many events are product type {most}?", 2 + PRODUCT_TYPES.index(most), sum(r["product_types"] == most for r in rows)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"trajectory_id": ex.trajectory_id, "question": ex.question, "template": ex.sql_template, "answer": min(ex.answer_value, 99)}


class QuestionVocab:
    def __init__(self, questions: list[str]):
        toks = ["[pad]", "[unk]"]
        for q in questions:
            for t in q.lower().replace("?", "").split():
                if t not in toks:
                    toks.append(t)
        self.itos = toks
        self.stoi = {t: i for i, t in enumerate(toks)}

    def encode(self, q: str, max_len: int = 16):
        ids = [self.stoi.get(t, 1) for t in q.lower().replace("?", "").split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids)


def collate(batch, vocab: QuestionVocab):
    return {
        "trajectory_id": torch.tensor([b["trajectory_id"] for b in batch]),
        "question_ids": torch.stack([vocab.encode(b["question"]) for b in batch]),
        "template": torch.tensor([b["template"] for b in batch]),
        "answer": torch.tensor([b["answer"] for b in batch]),
    }
