from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    epochs: int = 6
    lr: float = 1e-2
    batch_size: int = 32
    seed: int = 42


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_vocab(texts: list[str], max_vocab: int = 5000) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in texts:
        for tok in t.lower().split():
            counts[tok] = counts.get(tok, 0) + 1

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_vocab]
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in items:
        if w in vocab:
            continue
        vocab[w] = len(vocab)
    return vocab


def encode_batch(texts: list[str], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    ids: list[int] = []
    offsets: list[int] = [0]
    for t in texts:
        toks = t.lower().split()
        for tok in toks:
            ids.append(vocab.get(tok, 1))
        offsets.append(len(ids))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets[:-1], dtype=torch.long)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids, offsets)
        return self.fc(x).squeeze(-1)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def train_and_eval(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    cfg: TrainConfig | None = None,
    device: str | None = None,
) -> dict[str, float]:
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(cfg.seed)

    vocab = build_vocab(train_texts)
    model = TextClassifier(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    n = len(train_texts)
    idx = np.arange(n)
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for i in range(0, n, cfg.batch_size):
            batch_idx = idx[i : i + cfg.batch_size]
            texts = [train_texts[j] for j in batch_idx]
            labels = torch.tensor([train_labels[j] for j in batch_idx], dtype=torch.float32, device=device)
            ids, offsets = encode_batch(texts, vocab)
            ids, offsets = ids.to(device), offsets.to(device)
            logits = model(ids, offsets)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        ids, offsets = encode_batch(val_texts, vocab)
        ids, offsets = ids.to(device), offsets.to(device)
        logits = model(ids, offsets)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_true = np.asarray(val_labels, dtype=np.int64)
    y_pred = (probs >= 0.5).astype(np.int64)

    acc = float((y_true == y_pred).mean())
    f1 = _f1_score(y_true, y_pred)

    return {"acc": acc, "f1": f1}
