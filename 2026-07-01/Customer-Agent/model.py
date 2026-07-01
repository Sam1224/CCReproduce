from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PRODUCT_TYPES


class CustomerAgent(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 96, n_templates: int = 2 + len(PRODUCT_TYPES), max_answer: int = 100):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.template_head = nn.Linear(hidden, n_templates)
        self.answer_head = nn.Linear(hidden, max_answer)

    def encode(self, question_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(question_ids)
        _, h = self.encoder(x)
        return h[-1]

    def forward(self, question_ids: torch.Tensor):
        h = self.encode(question_ids)
        return self.template_head(h), self.answer_head(h)


def execute_template(data_dir: Path, trajectory_id: int, template_id: int) -> int:
    path = data_dir / f"trajectory_{trajectory_id}.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if template_id == 0:
        return sum(r["action_type"] == "purchase" for r in rows)
    if template_id == 1:
        return sum(r["action_type"] == "click" and float(r["price"]) > 500 for r in rows)
    product_type = PRODUCT_TYPES[template_id - 2]
    return sum(r["product_types"] == product_type for r in rows)


def sft_rlvr_loss(template_logits, answer_logits, template, answer):
    sft = F.cross_entropy(template_logits, template) + F.cross_entropy(answer_logits, answer)
    pred_answer = answer_logits.argmax(-1)
    verifiable_reward = (pred_answer == answer).float().mean()
    return sft - 0.05 * verifiable_reward
