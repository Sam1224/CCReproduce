from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    d_model: int = 96
    max_title_len: int = 8
    image_size: int = 64


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.gru = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)

    def forward(self, tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        outputs, _ = self.gru(x)
        denom = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (outputs * masks.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


class ImageEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, cfg.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.net(images).flatten(1)
        return self.proj(feat)


class SAMD2QModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg)
        self.image_encoder = ImageEncoder(cfg)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.term_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, title_tokens: torch.Tensor, title_masks: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        text_z = self.text_encoder(title_tokens, title_masks)
        image_z = self.image_encoder(images)
        fused = self.fusion(torch.cat([text_z, image_z], dim=-1))
        return self.term_head(fused)

    def predict_probs(self, title_tokens: torch.Tensor, title_masks: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(title_tokens, title_masks, images))

    def sample_term_sets(
        self,
        title_tokens: torch.Tensor,
        title_masks: torch.Tensor,
        images: torch.Tensor,
        vocab_terms: Sequence[str],
        topk: int = 5,
        samples_per_item: int = 4,
    ) -> Tuple[List[List[List[str]]], torch.Tensor]:
        logits = self.forward(title_tokens, title_masks, images)
        probs = torch.sigmoid(logits)
        batch_samples: List[List[List[str]]] = []
        log_prob_sums: List[torch.Tensor] = []
        for row in range(probs.size(0)):
            item_samples: List[List[str]] = []
            item_log_probs: List[torch.Tensor] = []
            row_probs = probs[row]
            top_values, top_indices = torch.topk(row_probs, k=topk * 2)
            for _ in range(samples_per_item):
                bernoulli = torch.bernoulli(top_values.clamp(min=0.05, max=0.95))
                if bernoulli.sum() == 0:
                    bernoulli[torch.argmax(top_values)] = 1.0
                chosen_indices = top_indices[bernoulli.bool()][:topk]
                terms = [vocab_terms[int(idx)] for idx in chosen_indices.tolist() if vocab_terms[int(idx)] not in {"<pad>", "<unk>"}]
                if not terms:
                    terms = [vocab_terms[int(top_indices[0])]]
                item_samples.append(terms)
                selected_probs = row_probs[chosen_indices].clamp(min=1e-5, max=1 - 1e-5)
                item_log_probs.append(torch.log(selected_probs).sum())
            batch_samples.append(item_samples)
            log_prob_sums.append(torch.stack(item_log_probs))
        return batch_samples, torch.stack(log_prob_sums)


def multilabel_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def compute_reward(
    sampled_terms: Sequence[str],
    gold_terms: torch.Tensor,
    visual_terms: torch.Tensor,
    commerce_weight: torch.Tensor,
    vocab_map: Dict[str, int],
) -> float:
    chosen = set(sampled_terms)
    if not chosen:
        return 0.0
    gold = {token for token, idx in vocab_map.items() if idx < gold_terms.numel() and gold_terms[idx] > 0.5}
    visual = {token for token, idx in vocab_map.items() if idx < visual_terms.numel() and visual_terms[idx] > 0.5}
    intersection = len(chosen.intersection(gold))
    precision = intersection / max(len(chosen), 1)
    recall = intersection / max(len(gold), 1)
    semantic = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    visual_bonus = len(chosen.intersection(visual)) / max(len(visual), 1)
    commercial = float(commerce_weight.item()) if intersection > 0 else 0.0
    return semantic + 0.35 * visual_bonus + 0.25 * commercial


def grpo_loss(
    log_prob_sums: torch.Tensor,
    sampled_terms: List[List[List[str]]],
    gold_terms: torch.Tensor,
    visual_terms: torch.Tensor,
    commerce_weights: torch.Tensor,
    vocab_map: Dict[str, int],
) -> torch.Tensor:
    batch_losses = []
    for row in range(log_prob_sums.size(0)):
        rewards = [
            compute_reward(sampled, gold_terms[row], visual_terms[row], commerce_weights[row], vocab_map)
            for sampled in sampled_terms[row]
        ]
        reward_tensor = torch.tensor(rewards, device=log_prob_sums.device, dtype=torch.float32)
        advantages = reward_tensor - reward_tensor.mean()
        batch_losses.append(-(advantages.detach() * log_prob_sums[row]).mean())
    return torch.stack(batch_losses).mean()


@torch.no_grad()
def decode_top_terms(
    probs: torch.Tensor,
    vocab_terms: Sequence[str],
    topk: int = 5,
    threshold: float = 0.35,
) -> List[List[str]]:
    decoded: List[List[str]] = []
    for row in probs:
        values, indices = torch.topk(row, k=topk)
        terms = [
            vocab_terms[int(idx)]
            for idx, value in zip(indices.tolist(), values.tolist())
            if value >= threshold and vocab_terms[int(idx)] not in {"<pad>", "<unk>"}
        ]
        if not terms:
            terms = [vocab_terms[int(indices[0])]]
        decoded.append(terms)
    return decoded
