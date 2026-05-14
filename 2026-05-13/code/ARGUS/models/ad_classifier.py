"""
Ad governance classifier with shared encoder.

Architecture (paper §3, inferred):
    Input: ad text + policy_tag context
    Encoder: shared transformer-like encoder
    Classifier head: binary (compliant / violation)

In production (paper): fine-tuned LLM (e.g., Qwen-7B)
In this toy: character n-gram encoder + MLP classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


LABEL_SPACE = ["compliant", "violation"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABEL_SPACE)}


@dataclass
class ClassifierOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    predicted_labels: list
    loss: Optional[torch.Tensor] = None


class TextEncoder(nn.Module):
    """Toy character n-gram bag-of-words encoder."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def _tokenize(self, texts: list):
        all_ids, offsets = [], []
        offset = 0
        for text in texts:
            ids = [hash(text[i:i+4]) % 10000 for i in range(0, min(len(text), 128), 2)]
            if not ids:
                ids = [0]
            all_ids.extend(ids)
            offsets.append(offset)
            offset += len(ids)
        return torch.tensor(all_ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)

    def forward(self, texts: list) -> torch.Tensor:
        ids, offsets = self._tokenize(texts)
        emb = self.embedding(ids, offsets)
        return self.proj(emb)


class AdClassifier(nn.Module):
    """
    Shared encoder + classification head for ad governance.

    Binary classification:
        P(violation | ad_text) = sigmoid(W * encoder(text))

    Loss (paper inferred, §3):
        L = BCE(logits, labels)

    With policy conditioning (paper §3.1, Policy Seeding):
        The encoder receives concatenated [ad_text; policy_description] as input.
        Here: simplified as ad_text only (policy context injected as text prefix).
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        texts: list,
        labels: Optional[torch.Tensor] = None,
    ) -> ClassifierOutput:
        hidden = self.encoder(texts)
        logits = self.classifier(hidden)
        probs = F.softmax(logits, dim=-1)
        predicted = [LABEL_SPACE[i] for i in logits.argmax(dim=-1).tolist()]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ClassifierOutput(
            logits=logits,
            probs=probs,
            predicted_labels=predicted,
            loss=loss,
        )

    def predict(self, text: str) -> tuple:
        self.eval()
        with torch.no_grad():
            out = self.forward([text])
        return out.predicted_labels[0], out.probs[0].max().item()
