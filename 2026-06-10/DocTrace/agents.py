import re
from dataclasses import dataclass

import numpy as np
import torch


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hash_vector(text: str, dim: int = 2048) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for tok in tokenize(text):
        idx = (hash(tok) % dim + dim) % dim
        vec[idx] += 1.0
    return vec


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


@dataclass
class EvidenceCandidate:
    pid: int
    text: str
    score: float
    reason: str


class RetrieverAgent:
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def retrieve(self, paragraphs: list[str], query: str, top_k: int = 8) -> list[EvidenceCandidate]:
        q = hash_vector(query, dim=self.dim)
        ps = torch.stack([hash_vector(p, dim=self.dim) for p in paragraphs], dim=0)
        sims = cosine_sim(ps, q.unsqueeze(0).expand_as(ps))
        order = torch.argsort(sims, descending=True)[:top_k]

        out = []
        for idx in order.tolist():
            out.append(
                EvidenceCandidate(
                    pid=idx,
                    text=paragraphs[idx],
                    score=float(sims[idx].item()),
                    reason="hash-cosine-sim",
                )
            )
        return out


class FilterModel(torch.nn.Module):
    def __init__(self, dim: int = 2048) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class FilterAgent:
    def __init__(self, model: FilterModel, dim: int = 2048, device: str = "cpu") -> None:
        self.model = model
        self.dim = dim
        self.device = device

    def score(self, query: str, paragraph: str) -> float:
        x = hash_vector(query + " [SEP] " + paragraph, dim=self.dim).to(self.device)
        with torch.no_grad():
            logit = self.model(x.unsqueeze(0)).squeeze(0)
            return float(torch.sigmoid(logit).item())

    def filter(self, candidates: list[EvidenceCandidate], query: str, top_k: int = 3) -> list[EvidenceCandidate]:
        m = re.search(r"entity_\d+", query)
        n = re.search(r"attr_\d+", query)
        entity = m.group(0) if m else ""
        attr = n.group(0) if n else ""

        scored = []
        for c in candidates:
            prob = self.score(query, c.text)
            match = (entity and entity in c.text) and (attr and attr in c.text)
            is_fact = c.text.lower().startswith("fact:")

            if match:
                final = 0.6 * prob + 0.4 * float(is_fact)
                reason = "match+fact+prob"
            else:
                final = 0.1 * prob
                reason = "prob-only"

            scored.append(
                EvidenceCandidate(
                    pid=c.pid,
                    text=c.text,
                    score=final,
                    reason=reason,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class AggregatorAgent:
    def aggregate(self, evidences: list[EvidenceCandidate]) -> str:
        return "\n".join([f"[P{e.pid}] {e.text}" for e in evidences])


class AnswerAgent:
    def answer(self, query: str, context: str) -> str:
        m = re.search(r"entity_(\d+)", query)
        n = re.search(r"attr_(\d+)", query)
        if not m or not n:
            return "UNKNOWN"
        entity = m.group(0)
        attr = n.group(0)

        pat = re.compile(rf"{re.escape(entity)}\s+{re.escape(attr)}\s+is\s+(value_\d+)", re.IGNORECASE)
        hit = pat.search(context)
        return hit.group(1) if hit else "UNKNOWN"


def token_cost(text: str) -> int:
    return len(tokenize(text))
