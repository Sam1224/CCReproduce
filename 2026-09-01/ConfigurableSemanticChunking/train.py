from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data import Document, QueryExample
from model import Chunk, ChunkScorer, Vocab, extra_features


class ChunkPairDataset(Dataset[Tuple[str, Chunk, float]]):
    def __init__(self, examples: Sequence[Tuple[str, Chunk, float]]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[str, Chunk, float]:
        return self.examples[index]


def make_training_pairs(queries: Sequence[QueryExample], chunks: Sequence[Chunk]) -> List[Tuple[str, Chunk, float]]:
    pairs: List[Tuple[str, Chunk, float]] = []
    for query in queries:
        for chunk in chunks:
            label = float(
                chunk.doc_id == query.expected_doc_id
                and (query.answer_relation == chunk.trigger or query.answer_relation == chunk.relation)
            )
            if label == 0.0 and query.expected_object.lower() in chunk.text.lower():
                label = 0.7
            pairs.append((query.query, chunk, label))
    return pairs


def collate(batch: Sequence[Tuple[str, Chunk, float]], vocab: Vocab):
    queries, chunks, labels = zip(*batch)
    query_ids = torch.stack([vocab.encode(query) for query in queries])
    chunk_ids = torch.stack([vocab.encode(chunk.text) for chunk in chunks])
    features = torch.tensor([extra_features(query, chunk) for query, chunk in zip(queries, chunks)], dtype=torch.float32)
    return query_ids, chunk_ids, features, torch.tensor(labels, dtype=torch.float32)


def train_scorer(
    documents: Sequence[Document],
    queries: Sequence[QueryExample],
    chunks: Sequence[Chunk],
    *,
    epochs: int = 120,
    lr: float = 3e-3,
) -> Tuple[ChunkScorer, Vocab]:
    texts = [doc.text for doc in documents] + [query.query for query in queries] + [chunk.text for chunk in chunks]
    vocab = Vocab.build(texts)
    model = ChunkScorer(vocab_size=len(vocab.token_to_id))
    dataset = ChunkPairDataset(make_training_pairs(queries, chunks))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate(batch, vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for query_ids, chunk_ids, features, labels in loader:
            logits = model(query_ids, chunk_ids, features)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, vocab


def rank_chunks(model: ChunkScorer, vocab: Vocab, query: str, chunks: Sequence[Chunk]) -> List[Tuple[Chunk, float]]:
    model.eval()
    with torch.no_grad():
        query_ids = torch.stack([vocab.encode(query) for _ in chunks])
        chunk_ids = torch.stack([vocab.encode(chunk.text) for chunk in chunks])
        features = torch.tensor([extra_features(query, chunk) for chunk in chunks], dtype=torch.float32)
        scores = torch.sigmoid(model(query_ids, chunk_ids, features)).tolist()
    return sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
