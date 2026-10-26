from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F

from data import RetrievalRecord


class SimpleTokenizer:
    def __init__(self) -> None:
        self.pad_id = 0
        self.unk_id = 1
        self.stoi = {"[PAD]": self.pad_id, "[UNK]": self.unk_id}

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.stoi:
                    self.stoi[token] = len(self.stoi)

    def encode(self, text: str, *, max_length: int = 16) -> List[int]:
        tokens = [self.stoi.get(token, self.unk_id) for token in self._tokenize(text)]
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            tokens = tokens + [self.pad_id] * (max_length - len(tokens))
        return tokens

    def batch_encode(self, texts: Iterable[str], *, max_length: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text, max_length=max_length) for text in texts]
        ids = torch.tensor(encoded, dtype=torch.long)
        mask = (ids != self.pad_id).long()
        return ids, mask

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


class SharedTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.output(pooled), dim=-1)


class TwoTowerRetriever(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.shared_encoder = SharedTextEncoder(vocab_size, hidden_size)

    def encode_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(input_ids, attention_mask)

    def encode_items(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(input_ids, attention_mask)

    def score(self, query_batch: tuple[torch.Tensor, torch.Tensor], item_batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q = self.encode_queries(*query_batch)
        i = self.encode_items(*item_batch)
        return (q * i).sum(dim=-1)


@dataclass
class AnnotationResult:
    grade: int
    trace: List[str]


class ToyCascadeAnnotator:
    """
    Toy replacement for:
    184M cross-encoder -> 2B LoRA LLM -> 8B LoRA LLM.

    The interface mirrors a calibrated cascade so the module can be replaced with
    real annotators while keeping the rest of the pipeline unchanged.
    """

    def annotate(self, record: RetrievalRecord) -> AnnotationResult:
        channels = set(record.channels)
        if record.lexical_overlap >= 0.88 and len(channels) >= 2:
            return AnnotationResult(4, ["cross_encoder_184m"])
        if record.lexical_overlap <= 0.12 and len(channels) == 1:
            return AnnotationResult(0, ["cross_encoder_184m"])
        if record.lexical_overlap >= 0.72 and ("lexical" in channels or "bm25" in channels):
            return AnnotationResult(3, ["cross_encoder_184m", "lora_llm_2b"])
        if record.lexical_overlap >= 0.45 or record.token_similarity >= 0.65:
            return AnnotationResult(2, ["cross_encoder_184m", "lora_llm_2b", "lora_llm_8b"])
        if record.lexical_overlap >= 0.22:
            return AnnotationResult(1, ["cross_encoder_184m", "lora_llm_2b", "lora_llm_8b"])
        return AnnotationResult(0, ["cross_encoder_184m", "lora_llm_2b", "lora_llm_8b"])

    def annotate_records(self, records: Iterable[RetrievalRecord]) -> List[RetrievalRecord]:
        annotated: List[RetrievalRecord] = []
        for record in records:
            result = self.annotate(record)
            record.teacher_grade = result.grade
            record.cascade_trace = result.trace
            annotated.append(record)
        return annotated


def bce_stage_loss(
    model: TwoTowerRetriever,
    tokenizer: SimpleTokenizer,
    positives: List[RetrievalRecord],
    negatives: List[RetrievalRecord],
) -> torch.Tensor:
    pairs = positives + negatives
    labels = torch.tensor([1.0] * len(positives) + [0.0] * len(negatives))
    query_batch = tokenizer.batch_encode([record.query for record in pairs])
    item_batch = tokenizer.batch_encode([record.item_title for record in pairs])
    logits = model.score(query_batch, item_batch)
    return F.binary_cross_entropy_with_logits(logits, labels)


def multiple_negative_ranking_loss(
    model: TwoTowerRetriever,
    tokenizer: SimpleTokenizer,
    triplets: List[tuple[RetrievalRecord, RetrievalRecord, RetrievalRecord]],
) -> torch.Tensor:
    if not triplets:
        return torch.tensor(0.0, requires_grad=True)
    queries = [triplet[0].query for triplet in triplets]
    positives = [triplet[1].item_title for triplet in triplets]
    negatives = [triplet[2].item_title for triplet in triplets]
    q_batch = tokenizer.batch_encode(queries)
    pos_batch = tokenizer.batch_encode(positives)
    neg_batch = tokenizer.batch_encode(negatives)
    q = model.encode_queries(*q_batch)
    pos = model.encode_items(*pos_batch)
    neg = model.encode_items(*neg_batch)
    logits = torch.stack([(q * pos).sum(-1), (q * neg).sum(-1)], dim=1)
    targets = torch.zeros(logits.size(0), dtype=torch.long)
    return F.cross_entropy(logits, targets)


def triplet_stage_loss(
    model: TwoTowerRetriever,
    tokenizer: SimpleTokenizer,
    triplets: List[tuple[RetrievalRecord, RetrievalRecord, RetrievalRecord]],
    margin: float = 0.2,
) -> torch.Tensor:
    if not triplets:
        return torch.tensor(0.0, requires_grad=True)
    queries = [triplet[0].query for triplet in triplets]
    positives = [triplet[1].item_title for triplet in triplets]
    negatives = [triplet[2].item_title for triplet in triplets]
    q_batch = tokenizer.batch_encode(queries)
    pos_batch = tokenizer.batch_encode(positives)
    neg_batch = tokenizer.batch_encode(negatives)
    q = model.encode_queries(*q_batch)
    pos = model.encode_items(*pos_batch)
    neg = model.encode_items(*neg_batch)
    pos_scores = (q * pos).sum(-1)
    neg_scores = (q * neg).sum(-1)
    return F.relu(margin - pos_scores + neg_scores).mean()


def ndcg_at_k(scores: List[float], grades: List[int], k: int = 3) -> float:
    ranked = sorted(zip(scores, grades), key=lambda item: item[0], reverse=True)[:k]
    dcg = 0.0
    for index, (_, grade) in enumerate(ranked):
        dcg += (2**grade - 1) / math.log2(index + 2)
    ideal = sorted(grades, reverse=True)[:k]
    idcg = 0.0
    for index, grade in enumerate(ideal):
        idcg += (2**grade - 1) / math.log2(index + 2)
    return dcg / idcg if idcg > 0 else 0.0
