from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Article:
    title: str
    body: str
    language: str


@dataclass
class Video:
    video_id: str
    title: str
    description: str
    language: str
    freshness: float = 0.0
    performance: float = 0.0


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        vectors = self.embedding(token_ids, offsets)
        return F.normalize(self.projection(vectors), dim=-1)


class ContextualVideoMatcher(nn.Module):
    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embedding_dim)
        self.score_weights = nn.Parameter(torch.tensor([1.0, 0.05, 0.05]))

    def encode_texts(self, tokenized_texts: Sequence[Sequence[int]]) -> torch.Tensor:
        flat_tokens: List[int] = []
        offsets: List[int] = []
        cursor = 0
        for tokens in tokenized_texts:
            offsets.append(cursor)
            flat_tokens.extend(tokens or [0])
            cursor += max(len(tokens), 1)
        token_tensor = torch.tensor(flat_tokens, dtype=torch.long)
        offset_tensor = torch.tensor(offsets, dtype=torch.long)
        return self.encoder(token_tensor, offset_tensor)

    def score_candidates(self, article_tokens: Sequence[int], video_tokens: Sequence[Sequence[int]], freshness: torch.Tensor, performance: torch.Tensor) -> torch.Tensor:
        article_vector = self.encode_texts([article_tokens])
        video_vectors = self.encode_texts(video_tokens)
        similarity = torch.matmul(video_vectors, article_vector.squeeze(0))
        feature_matrix = torch.stack([similarity, freshness, performance], dim=-1)
        return torch.matmul(feature_matrix, self.score_weights)


def synthesize_hypothetical_video_metadata(article: Article) -> str:
    clean_body = " ".join(article.body.split())
    summary = clean_body[:320]
    return f"{article.title}. Relevant video about {summary}. Language: {article.language}."


def tokenize(text: str, vocab_size: int = 4096) -> List[int]:
    return [abs(hash(token.lower())) % vocab_size for token in text.split()]


def retrieve_videos(model: ContextualVideoMatcher, article: Article, videos: Sequence[Video], threshold: float = 0.15, top_k: int = 3) -> List[Tuple[Video, float]]:
    article_query = synthesize_hypothetical_video_metadata(article)
    same_language_videos = [video for video in videos if video.language == article.language]
    if not same_language_videos:
        return []
    video_texts = [f"{video.title}. {video.description}" for video in same_language_videos]
    scores = model.score_candidates(
        tokenize(article_query),
        [tokenize(text) for text in video_texts],
        torch.tensor([video.freshness for video in same_language_videos], dtype=torch.float),
        torch.tensor([video.performance for video in same_language_videos], dtype=torch.float),
    )
    ranked = sorted(zip(same_language_videos, scores.detach().tolist()), key=lambda item: item[1], reverse=True)
    return [(video, score) for video, score in ranked[:top_k] if score >= threshold]


def contrastive_training_loss(article_vectors: torch.Tensor, positive_video_vectors: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = torch.matmul(article_vectors, positive_video_vectors.T) / temperature
    labels = torch.arange(article_vectors.size(0), device=article_vectors.device)
    return F.cross_entropy(logits, labels)
