from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PlanPage


@dataclass
class RetrievalOutput:
    scores: torch.Tensor
    topk_ids: torch.Tensor
    maxsim_heatmap: torch.Tensor


class MultiVectorRetriever(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.patch_encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, query_vec: torch.Tensor, pages: List[PlanPage], topk: int = 5) -> RetrievalOutput:
        q = F.normalize(self.query_encoder(query_vec), dim=-1)
        page_scores = []
        heatmaps = []
        for page in pages:
            patches = F.normalize(self.patch_encoder(page.patch_features.to(query_vec.device)), dim=-1)
            sim = q @ patches.t()
            heatmaps.append(sim)
            page_scores.append(sim.max(dim=-1).values)
        scores = torch.stack(page_scores, dim=1)
        top = torch.topk(scores, k=topk, dim=1).indices
        heatmap = torch.stack(heatmaps, dim=1)
        return RetrievalOutput(scores=scores, topk_ids=top, maxsim_heatmap=heatmap)


class ComplianceAuditor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(0.05))

    def forward(self, retrieved_rule_value: torch.Tensor, proposed_value: torch.Tensor) -> torch.Tensor:
        diff = retrieved_rule_value - proposed_value
        pass_logit = diff - self.margin.abs()
        fail_logit = -diff + self.margin.abs()
        return torch.stack([fail_logit, pass_logit], dim=-1)
