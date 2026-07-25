from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EAACDConfig:
    vocab_size: int
    hidden_size: int = 96
    num_experts: int = 6
    top_k: int = 2
    high_layer_start: int = 1
    contrast_alpha: float = 0.65
    amplification_beta: float = 0.35
    consistency_weight: float = 0.5


class SparseMoELayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.top_k = top_k

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(hidden)
        router_probs = F.softmax(router_logits, dim=-1)
        top_values, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=2)
        mask = torch.zeros_like(router_probs).scatter_(-1, top_indices, top_values)
        mask = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed = torch.einsum("bse,bseh->bsh", mask, expert_outputs)
        return hidden + mixed, router_probs, expert_outputs


class ToyMoELanguageModel(nn.Module):
    def __init__(self, config: EAACDConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SparseMoELayer(config.hidden_size, config.num_experts, config.top_k) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.factual_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.embedding(input_ids)
        router_history = []
        expert_history = []
        for layer in self.layers:
            hidden, router_probs, expert_outputs = layer(hidden)
            router_history.append(router_probs)
            expert_history.append(expert_outputs)
        pooled = self.norm(hidden).mean(dim=1)
        logits = self.output(pooled)
        factual_logit = self.factual_head(pooled).squeeze(-1)
        return {
            "logits": logits,
            "factual_logit": factual_logit,
            "router_probs": torch.stack(router_history, dim=1),
            "expert_outputs": torch.stack(expert_history, dim=1),
            "pooled": pooled,
        }


class EAACDDecoder(nn.Module):
    def __init__(self, model: ToyMoELanguageModel, config: EAACDConfig):
        super().__init__()
        self.model = model
        self.config = config

    def expert_reliability(self, router_probs: torch.Tensor) -> torch.Tensor:
        high_probs = router_probs[:, self.config.high_layer_start :, :, :].mean(dim=(1, 2))
        confidence = high_probs / high_probs.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        consistency = 1.0 - high_probs.std(dim=-1, keepdim=True).expand_as(high_probs)
        reliability = confidence + self.config.consistency_weight * consistency
        return reliability

    def contrast_logits(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_ids)
        base_logits = outputs["logits"]
        reliability = self.expert_reliability(outputs["router_probs"])
        high_index = reliability.argmax(dim=-1)
        low_mask = F.one_hot(high_index, reliability.size(-1)).bool().logical_not()
        negative_strength = (outputs["router_probs"][:, self.config.high_layer_start :, :, :] * low_mask[:, None, None, :]).sum(dim=-1).mean(dim=(1, 2))
        factual_prob = torch.sigmoid(outputs["factual_logit"])
        adaptive_alpha = self.config.contrast_alpha * (1.0 - factual_prob + negative_strength).unsqueeze(-1)
        amplified_negative = self.config.amplification_beta * negative_strength.unsqueeze(-1) * torch.tanh(base_logits)
        calibrated_logits = base_logits + adaptive_alpha * base_logits - amplified_negative
        return {
            "logits": calibrated_logits,
            "base_logits": base_logits,
            "reliability": reliability,
            "negative_strength": negative_strength,
            "factual_prob": factual_prob,
        }

    @torch.no_grad()
    def generate_answer_ids(self, input_ids: torch.Tensor, max_new_tokens: int = 4) -> torch.Tensor:
        generated = []
        current = input_ids
        for _ in range(max_new_tokens):
            decoded = self.contrast_logits(current)
            next_token = decoded["logits"].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            current = torch.cat([current[:, 1:], next_token], dim=1)
        return torch.cat(generated, dim=1)
