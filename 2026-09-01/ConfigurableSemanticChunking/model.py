from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class SemanticChunkConfig:
    max_window_sentences: int = 2
    context_sentences: int = 1
    primary_triggers: Tuple[str, ...] = ("claims", "flags", "requires", "receives", "extracts", "maps")
    secondary_triggers: Tuple[str, ...] = ("matches", "rewritten", "hurts", "improve", "reduces")
    relation_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "moderation": 4.0,
            "policy": 3.5,
            "claim": 3.0,
            "similarity": 2.8,
            "attribute": 2.5,
            "normalization": 2.5,
            "promotion": 2.0,
            "correction": 1.8,
            "quality": 1.5,
        }
    )
    entity_bonus: float = 0.6
    proposition_bonus: float = 1.2


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str
    sentence_ids: Tuple[int, ...]
    relation: str
    trigger: str
    entities: Tuple[str, ...]
    score: float


class SemanticChunker:
    def __init__(self, config: SemanticChunkConfig, relation_map: Dict[str, str], entities: Sequence[str]):
        self.config = config
        self.relation_map = {key.lower(): value for key, value in relation_map.items()}
        self.entities = tuple(entities)

    def split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def find_entities(self, text: str) -> Tuple[str, ...]:
        lowered = text.lower()
        return tuple(entity for entity in self.entities if entity.lower() in lowered)

    def find_triggers(self, text: str) -> List[str]:
        tokens = set(tokenize(text))
        return [trigger for trigger in self.config.primary_triggers + self.config.secondary_triggers if trigger.lower() in tokens]

    def trigger_tier_bonus(self, trigger: str) -> float:
        if trigger in self.config.primary_triggers:
            return 1.0
        if trigger in self.config.secondary_triggers:
            return 0.5
        return 0.0

    def extract_proposition(self, sentence: str, trigger: str) -> str:
        tokens = sentence.split()
        trigger_index = next((idx for idx, token in enumerate(tokens) if trigger.lower() in token.lower().strip(".,;:")), -1)
        if trigger_index < 0:
            return sentence
        start = max(0, trigger_index - 5)
        end = min(len(tokens), trigger_index + 8)
        return " ".join(tokens[start:end]).strip()

    def score_candidate(self, text: str, trigger: str, relation: str, entities: Sequence[str], proposition: bool) -> float:
        relation_score = self.config.relation_weights.get(relation, 1.0)
        entity_score = self.config.entity_bonus * len(entities)
        trigger_score = self.trigger_tier_bonus(trigger)
        proposition_score = self.config.proposition_bonus if proposition else 0.0
        length_penalty = 0.02 * max(0, len(tokenize(text)) - 32)
        return relation_score + entity_score + trigger_score + proposition_score - length_penalty

    def chunk_document(self, doc_id: str, text: str) -> List[Chunk]:
        sentences = self.split_sentences(text)
        chunks: List[Chunk] = []
        for sent_idx, sentence in enumerate(sentences):
            triggers = self.find_triggers(sentence)
            if not triggers:
                continue
            for trigger in triggers:
                start = max(0, sent_idx - self.config.context_sentences)
                end = min(len(sentences), sent_idx + self.config.context_sentences + 1)
                window_sentences = sentences[start:end][: self.config.max_window_sentences]
                window_text = " ".join(window_sentences)
                proposition = self.extract_proposition(sentence, trigger)
                relation = self.relation_map.get(trigger.lower(), "other")
                for candidate_text, is_proposition in ((window_text, False), (proposition, True)):
                    entities = self.find_entities(candidate_text)
                    score = self.score_candidate(candidate_text, trigger, relation, entities, is_proposition)
                    chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            text=candidate_text,
                            sentence_ids=tuple(range(start, min(end, start + len(window_sentences)))),
                            relation=relation,
                            trigger=trigger,
                            entities=entities,
                            score=score,
                        )
                    )
        return self.resolve(chunks)

    def resolve(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        best: Dict[Tuple[str, str, str], Chunk] = {}
        for chunk in chunks:
            key = (chunk.doc_id, chunk.relation, " ".join(tokenize(chunk.text)[:10]))
            current = best.get(key)
            if current is None or chunk.score > current.score:
                best[key] = chunk
        return sorted(best.values(), key=lambda item: item.score, reverse=True)


class FixedSizeChunker:
    def __init__(self, window_tokens: int = 12):
        self.window_tokens = window_tokens

    def chunk_document(self, doc_id: str, text: str) -> List[Chunk]:
        words = text.split()
        chunks: List[Chunk] = []
        for start in range(0, len(words), self.window_tokens):
            segment = " ".join(words[start : start + self.window_tokens])
            chunks.append(Chunk(doc_id, segment, (start,), "fixed", "", tuple(), 0.0))
        return chunks


class Vocab:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id

    @classmethod
    def build(cls, texts: Iterable[str]) -> "Vocab":
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for text in texts:
            for token in tokenize(text):
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_len: int = 96) -> torch.Tensor:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)[:max_len]]
        ids.extend([0] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)


class ChunkScorer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(dim * 4 + 3, dim), nn.ReLU(), nn.Linear(dim, 1))

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).float().unsqueeze(-1)
        embedded = self.embedding(input_ids)
        return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, query_ids: torch.Tensor, chunk_ids: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        query_vec = self.encode_text(query_ids)
        chunk_vec = self.encode_text(chunk_ids)
        features = torch.cat([query_vec, chunk_vec, query_vec * chunk_vec, torch.abs(query_vec - chunk_vec), extra_features], dim=-1)
        return self.proj(features).squeeze(-1)


def lexical_overlap(query: str, chunk: str) -> float:
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(chunk))
    if not query_tokens or not chunk_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / math.sqrt(len(query_tokens) * len(chunk_tokens))


def extra_features(query: str, chunk: Chunk) -> List[float]:
    return [lexical_overlap(query, chunk.text), chunk.score / 10.0, min(len(chunk.entities), 4) / 4.0]
