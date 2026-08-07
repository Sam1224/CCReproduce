from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from dataset import HistoricalChunk, RequestCase
from experience_tree import ExperienceTree

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_\.]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Vocab:
    def __init__(self, texts: Iterable[str]) -> None:
        ordered = ["<pad>"]
        for text in texts:
            for token in tokenize(text):
                if token not in ordered:
                    ordered.append(token)
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.id_to_token = ordered

    def encode(self, text: str) -> List[int]:
        tokens = tokenize(text)
        if not tokens:
            return [0]
        return [self.token_to_id.get(token, 0) for token in tokens]

    def __len__(self) -> int:
        return len(self.id_to_token)


class DenseRetriever(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(token_ids, offsets)
        projected = self.proj(embedding)
        return nn.functional.normalize(projected, dim=-1)


class Reranker(nn.Module):
    def __init__(self, input_dim: int = 6) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


@dataclass
class RetrievalPair:
    request: RequestCase
    chunk: HistoricalChunk
    label: float


def build_pairs(requests: Sequence[RequestCase], chunks: Sequence[HistoricalChunk]) -> List[RetrievalPair]:
    pairs: List[RetrievalPair] = []
    for request in requests:
        for chunk in chunks:
            label = 0.0
            if request.scenario == chunk.scenario and request.objective == chunk.objective:
                label = 1.0
            elif request.objective == chunk.objective:
                label = 0.5
            pairs.append(RetrievalPair(request=request, chunk=chunk, label=label))
    return pairs


def pack_texts(vocab: Vocab, texts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ids: List[int] = []
    offsets: List[int] = []
    offset = 0
    for text in texts:
        encoded = vocab.encode(text)
        offsets.append(offset)
        ids.extend(encoded)
        offset += len(encoded)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(offsets, dtype=torch.long, device=device)


@torch.no_grad()
def dense_similarity(model: DenseRetriever, vocab: Vocab, left_texts: Sequence[str], right_texts: Sequence[str], device: torch.device) -> np.ndarray:
    left_ids, left_offsets = pack_texts(vocab, left_texts, device)
    right_ids, right_offsets = pack_texts(vocab, right_texts, device)
    left_vec = model(left_ids, left_offsets)
    right_vec = model(right_ids, right_offsets)
    scores = left_vec @ right_vec.t()
    return scores.cpu().numpy()


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[HistoricalChunk],
        tree: ExperienceTree,
        vocab: Vocab,
        dense_model: DenseRetriever,
        reranker: Reranker,
        vectorizer: TfidfVectorizer,
        chunk_matrix,
        device: torch.device,
    ) -> None:
        self.chunks = list(chunks)
        self.tree = tree
        self.vocab = vocab
        self.dense_model = dense_model
        self.reranker = reranker
        self.vectorizer = vectorizer
        self.chunk_matrix = chunk_matrix
        self.device = device

    def retrieve(self, request: RequestCase, topk: int = 8) -> List[dict]:
        request_sparse = self.vectorizer.transform([request.text])
        sparse_scores = (request_sparse @ self.chunk_matrix.T).toarray().ravel()
        dense_scores = dense_similarity(self.dense_model, self.vocab, [request.text], [chunk.text for chunk in self.chunks], self.device).ravel()
        candidate_rows = []
        traffic_max = max(chunk.traffic for chunk in self.chunks)
        for index, chunk in enumerate(self.chunks):
            tree_boost = self.tree.tree_boost(request, chunk)
            features = np.array(
                [
                    sparse_scores[index],
                    dense_scores[index],
                    tree_boost,
                    float(request.scenario == chunk.scenario),
                    float(request.stage == chunk.stage),
                    chunk.traffic / traffic_max,
                ],
                dtype=np.float32,
            )
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            rerank_score = float(self.reranker(feature_tensor).item())
            candidate_rows.append(
                {
                    "chunk": chunk,
                    "sparse_score": float(sparse_scores[index]),
                    "dense_score": float(dense_scores[index]),
                    "tree_boost": float(tree_boost),
                    "score": float(rerank_score + 0.3 * sparse_scores[index] + 0.3 * dense_scores[index] + 0.2 * tree_boost),
                }
            )
        candidate_rows.sort(key=lambda row: row["score"], reverse=True)
        return candidate_rows[:topk]
