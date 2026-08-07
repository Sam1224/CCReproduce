from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from dataset import HistoricalChunk, RequestCase, bootstrap_data
from experience_tree import ExperienceTree
from model import MECHANISM_TO_ID, StrategyValueNet, ValueNetConfig, request_to_tensor
from retriever import DenseRetriever, Reranker, Vocab, build_pairs, pack_texts


def train_dense_model(model: DenseRetriever, vocab: Vocab, requests: List[RequestCase], chunks: List[HistoricalChunk], device: torch.device) -> None:
    pairs = build_pairs(requests, chunks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        request_ids, request_offsets = pack_texts(vocab, [pair.request.text for pair in pairs], device)
        chunk_ids, chunk_offsets = pack_texts(vocab, [pair.chunk.text for pair in pairs], device)
        request_vec = model(request_ids, request_offsets)
        chunk_vec = model(chunk_ids, chunk_offsets)
        logits = (request_vec * chunk_vec).sum(dim=-1)
        labels = torch.tensor([pair.label for pair in pairs], dtype=torch.float32, device=device)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()


def train_reranker(reranker: Reranker, requests: List[RequestCase], chunks: List[HistoricalChunk], tree: ExperienceTree, vectorizer: TfidfVectorizer, dense_model: DenseRetriever, vocab: Vocab, device: torch.device) -> None:
    pairs = build_pairs(requests, chunks)
    chunk_matrix = vectorizer.transform([chunk.text for chunk in chunks])
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    dense_model.eval()
    for _ in range(50):
        features = []
        labels = []
        for pair in pairs:
            request_sparse = vectorizer.transform([pair.request.text])
            sparse_score = float((request_sparse @ chunk_matrix[chunks.index(pair.chunk)].T).toarray().item())
            request_ids, request_offsets = pack_texts(vocab, [pair.request.text], device)
            chunk_ids, chunk_offsets = pack_texts(vocab, [pair.chunk.text], device)
            with torch.no_grad():
                request_vec = dense_model(request_ids, request_offsets)
                chunk_vec = dense_model(chunk_ids, chunk_offsets)
                dense_score = float((request_vec * chunk_vec).sum(dim=-1).item())
            tree_boost = tree.tree_boost(pair.request, pair.chunk)
            features.append(
                [
                    sparse_score,
                    dense_score,
                    tree_boost,
                    float(pair.request.scenario == pair.chunk.scenario),
                    float(pair.request.stage == pair.chunk.stage),
                    pair.chunk.traffic / 150000.0,
                ]
            )
            labels.append(pair.label)
        feature_tensor = torch.tensor(np.array(features), dtype=torch.float32, device=device)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = reranker(feature_tensor)
        loss = loss_fn(logits, label_tensor)
        loss.backward()
        optimizer.step()


def train_value_net(value_net: StrategyValueNet, chunks: List[HistoricalChunk], requests: List[RequestCase], device: torch.device) -> None:
    request_map = {(request.scenario, request.objective): request for request in requests}

    def resolve_request(chunk: HistoricalChunk) -> RequestCase:
        exact = request_map.get((chunk.scenario, chunk.objective))
        if exact is not None:
            return exact
        for request in requests:
            if request.objective == chunk.objective:
                return request
        for request in requests:
            if request.scenario == chunk.scenario:
                return request
        return requests[0]

    aligned_requests = [resolve_request(chunk) for chunk in chunks]
    request_tensor = request_to_tensor(aligned_requests, device)
    mechanism_ids = torch.tensor([MECHANISM_TO_ID[chunk.mechanism] for chunk in chunks], dtype=torch.long, device=device)
    params = torch.tensor([chunk.params for chunk in chunks], dtype=torch.float32, device=device)
    targets = torch.tensor(
        [[chunk.observed["gmv"], chunk.observed["ctr"], chunk.observed["refund_safety"], chunk.observed["creator_fairness"]] for chunk in chunks],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(value_net.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        predictions = value_net(request_tensor, mechanism_ids, params)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks, requests = bootstrap_data(data_dir)
    tree = ExperienceTree(chunks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab([chunk.text for chunk in chunks] + [request.text for request in requests])
    dense_model = DenseRetriever(len(vocab)).to(device)
    reranker = Reranker().to(device)
    value_net = StrategyValueNet(ValueNetConfig()).to(device)

    train_dense_model(dense_model, vocab, requests, chunks, device)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit([chunk.text for chunk in chunks] + [request.text for request in requests])
    train_reranker(reranker, requests, chunks, tree, vectorizer, dense_model, vocab, device)
    train_value_net(value_net, chunks, requests, device)

    torch.save(
        {
            "dense_model": dense_model.state_dict(),
            "reranker": reranker.state_dict(),
            "value_net": value_net.state_dict(),
            "vocab": vocab,
            "vectorizer": vectorizer,
            "chunks": chunks,
            "requests": requests,
        },
        output_dir / "ab_agent.pt",
    )
    print(f"Saved checkpoint to {output_dir / 'ab_agent.pt'}")


if __name__ == "__main__":
    main()
