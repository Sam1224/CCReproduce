from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class ReWEIGHConfig:
    alpha: float = 2.5
    topk_reference: int = 512
    stability_max_std: float = 0.2
    min_reference: float = 0.0
    eps: float = 1e-8


@dataclass
class ReWEIGHState:
    reference: torch.Tensor
    stable_mask: torch.Tensor


class ReWEIGHCalibrator:
    def __init__(self, config: Optional[ReWEIGHConfig] = None):
        self.config = config or ReWEIGHConfig()

    @staticmethod
    def dense_mean_reciprocal_rank(visual_logits: torch.Tensor) -> torch.Tensor:
        if visual_logits.dim() != 3:
            raise ValueError("visual_logits must have shape [batch, visual_tokens, vocab_size]")
        order = visual_logits.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        rank_values = torch.arange(1, visual_logits.size(-1) + 1, device=visual_logits.device, dtype=torch.float32)
        ranks.scatter_(-1, order, rank_values.expand_as(order))
        return (1.0 / ranks).mean(dim=1)

    def fit(self, evidence_batches: Iterable[torch.Tensor]) -> ReWEIGHState:
        evidences = []
        for visual_logits in evidence_batches:
            evidences.append(self.dense_mean_reciprocal_rank(visual_logits).detach().cpu())
        if not evidences:
            raise ValueError("at least one calibration batch is required")
        evidence = torch.cat(evidences, dim=0)
        reference = evidence.mean(dim=0)
        std = evidence.std(dim=0, unbiased=False)
        topk_mask = torch.zeros_like(reference, dtype=torch.bool)
        topk = min(self.config.topk_reference, reference.numel())
        topk_mask[reference.topk(topk).indices] = True
        stable_mask = topk_mask & (std <= self.config.stability_max_std) & (reference >= self.config.min_reference)
        return ReWEIGHState(reference=reference, stable_mask=stable_mask)


class ReWEIGHLogitsProcessor:
    def __init__(self, state: ReWEIGHState, config: Optional[ReWEIGHConfig] = None):
        self.state = state
        self.config = config or ReWEIGHConfig()

    def __call__(self, next_token_logits: torch.Tensor, image_evidence: torch.Tensor) -> torch.Tensor:
        reference = self.state.reference.to(next_token_logits.device)
        stable_mask = self.state.stable_mask.to(next_token_logits.device)
        if image_evidence.dim() == 1:
            image_evidence = image_evidence.unsqueeze(0)
        reference = reference.unsqueeze(0).expand_as(next_token_logits)
        stable_mask = stable_mask.unsqueeze(0).expand_as(next_token_logits)
        shortfall = (reference - image_evidence) / reference.clamp_min(self.config.eps)
        penalty = self.config.alpha * shortfall.clamp(min=0.0, max=1.0)
        penalty = torch.where(stable_mask, penalty, torch.zeros_like(penalty))
        return next_token_logits - penalty


def greedy_decode_with_reweigh(model, image: torch.Tensor, prompt_ids: torch.Tensor, state: ReWEIGHState, max_new_tokens: int = 8, config: Optional[ReWEIGHConfig] = None) -> torch.Tensor:
    processor = ReWEIGHLogitsProcessor(state, config)
    generated = prompt_ids.clone()
    with torch.no_grad():
        outputs = model(image, generated)
        image_evidence = ReWEIGHCalibrator.dense_mean_reciprocal_rank(model.visual_readout(outputs["visual_hidden_states"]))
        for _ in range(max_new_tokens):
            outputs = model(image, generated)
            logits = processor(outputs["logits"][:, -1, :], image_evidence)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
    return generated
