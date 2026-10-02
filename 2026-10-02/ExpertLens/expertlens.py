from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalMoE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_experts)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        hidden = torch.tanh(self.input_proj(x))
        router_logits = self.router(hidden)
        weights = F.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=1)
        mixed = torch.einsum("be,beh->bh", weights, expert_outputs)
        return self.classifier(mixed), weights


@dataclass
class LensResult:
    expert_id: int
    tokens: List[str]
    score: float


class ExpertLens:
    def __init__(self, model: MultimodalMoE, vocab_tokens: List[str], token_embeddings: torch.Tensor):
        self.model = model
        self.vocab_tokens = vocab_tokens
        self.token_embeddings = F.normalize(token_embeddings, dim=-1)

    def decode_expert(self, expert_id: int, top_k: int = 5) -> LensResult:
        router_direction = self.model.router.weight[expert_id].detach()
        expert_first_layer = self.model.experts[expert_id][0].weight.detach().mean(dim=0)
        semantic_direction = F.normalize(router_direction + expert_first_layer, dim=0)
        scores = self.token_embeddings @ semantic_direction
        top_scores, top_ids = torch.topk(scores, k=top_k)
        return LensResult(
            expert_id=expert_id,
            tokens=[self.vocab_tokens[index] for index in top_ids.tolist()],
            score=float(top_scores.mean()),
        )

    def select_domain_experts(self, domain_keywords: Iterable[str], min_overlap: int = 1) -> Set[int]:
        keywords = set(domain_keywords)
        selected = set()
        for expert_id in range(len(self.model.experts)):
            decoded = self.decode_expert(expert_id, top_k=6)
            if len(keywords.intersection(decoded.tokens)) >= min_overlap:
                selected.add(expert_id)
        if not selected:
            selected.add(max(range(len(self.model.experts)), key=lambda idx: self.decode_expert(idx).score))
        return selected


def make_token_embeddings(vocab: List[str], hidden_dim: int, domain_keywords: Dict[str, List[str]]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(len(vocab), hidden_dim, generator=generator) * 0.15
    for domain_index, keywords in enumerate(domain_keywords.values()):
        base_direction = torch.zeros(hidden_dim)
        base_direction[domain_index * 4:domain_index * 4 + 4] = 1.0
        for keyword in keywords:
            base[vocab.index(keyword)] += base_direction
    return base


def freeze_for_selected_experts(model: MultimodalMoE, selected_experts: Set[int]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for expert_id in selected_experts:
        for parameter in model.experts[expert_id].parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def train_epoch(model: MultimodalMoE, batches, lr: float = 1e-2) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    total_loss = 0.0
    for x, y in batches:
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / max(1, len(batches))


def accuracy(model: MultimodalMoE, batches) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            pred = model(x)[0].argmax(dim=-1)
            correct += int((pred == y).sum())
            total += y.numel()
    return correct / max(1, total)
