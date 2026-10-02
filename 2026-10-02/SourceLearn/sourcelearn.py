import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\-]+", text.lower())


class Vocabulary:
    def __init__(self, texts: Iterable[str]):
        tokens = sorted({token for text in texts for token in tokenize(text)})
        self.stoi = {token: idx for idx, token in enumerate(tokens)}

    def encode(self, text: str) -> torch.Tensor:
        vector = torch.zeros(len(self.stoi), dtype=torch.float32)
        for token in tokenize(text):
            if token in self.stoi:
                vector[self.stoi[token]] += 1.0
        return vector / vector.norm().clamp_min(1.0)


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(bag), dim=-1)


@dataclass
class SourceModel:
    concepts: Dict[str, str] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    gaps: List[str] = field(default_factory=list)

    def update_from_source(self, doc: Dict[str, object], reason: str) -> None:
        for entity in doc["entities"]:
            old = self.concepts.get(entity, "")
            addition = f"{entity}: {doc['text']} [{reason}]"
            self.concepts[entity] = addition if not old else old + "\n" + addition
            self.confidence[entity] = min(1.0, self.confidence.get(entity, 0.2) + 0.25)

    def mark_gap(self, gap: str) -> None:
        if gap not in self.gaps:
            self.gaps.append(gap)


class SourceLearn:
    def __init__(self, docs: List[Dict[str, object]], vocab: Vocabulary, hidden_size: int = 64):
        self.docs = docs
        self.vocab = vocab
        self.encoder = TinyEncoder(len(vocab.stoi), hidden_size)
        self.source_model = SourceModel()

    def doc_tensor(self) -> torch.Tensor:
        return torch.stack([self.vocab.encode(str(doc["text"])) for doc in self.docs])

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[Dict[str, object], float]]:
        with torch.no_grad():
            query_emb = self.encoder(self.vocab.encode(query).unsqueeze(0))
            doc_emb = self.encoder(self.doc_tensor())
            scores = (query_emb @ doc_emb.T).squeeze(0)
            indexes = scores.argsort(descending=True)[:top_k].tolist()
        return [(self.docs[index], float(scores[index])) for index in indexes]

    def answer(self, question: str) -> str:
        doc, _ = self.retrieve(question, 1)[0]
        snippets = []
        for entity in doc["entities"]:
            if entity in self.source_model.concepts:
                snippets.append(self.source_model.concepts[entity].split("\n")[-1])
        if not snippets:
            self.source_model.mark_gap(str(doc["id"]))
            return str(doc["text"])
        return " ".join(snippets[:2])

    def self_directed_source_learning(self) -> None:
        for doc in self.docs:
            needs_study = any(self.source_model.confidence.get(entity, 0.0) < 0.5 for entity in doc["entities"])
            if needs_study or doc["id"] in self.source_model.gaps:
                self.source_model.update_from_source(doc, reason="self-directed")

    def task_guided_source_learning(self, task: Dict[str, str]) -> bool:
        retrieved = self.retrieve(task["question"], 1)[0][0]
        success = retrieved["id"] == task["gold_doc"]
        if success:
            self.source_model.update_from_source(retrieved, reason="task-guided-success")
        else:
            self.source_model.mark_gap(task["gold_doc"])
            for doc in self.docs:
                if doc["id"] == task["gold_doc"]:
                    self.source_model.update_from_source(doc, reason="task-guided-correction")
        return success


def train_retriever(model: SourceLearn, tasks: List[Dict[str, str]], epochs: int = 120, lr: float = 2e-2) -> None:
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    doc_bags = model.doc_tensor()
    doc_ids = [doc["id"] for doc in model.docs]
    gold = torch.tensor([doc_ids.index(task["gold_doc"]) for task in tasks], dtype=torch.long)
    query_bags = torch.stack([model.vocab.encode(task["question"]) for task in tasks])
    for _ in range(epochs):
        query_emb = model.encoder(query_bags)
        doc_emb = model.encoder(doc_bags)
        logits = query_emb @ doc_emb.T / math.sqrt(query_emb.shape[-1])
        loss = F.cross_entropy(logits, gold)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_retrieval(model: SourceLearn, tasks: List[Dict[str, str]]) -> float:
    correct = 0
    for task in tasks:
        retrieved = model.retrieve(task["question"], 1)[0][0]
        correct += int(retrieved["id"] == task["gold_doc"])
    return correct / max(1, len(tasks))
