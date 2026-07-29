from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisPlanner(nn.Module):
    def __init__(self, profile_dim: int = 40, embed_dim: int = 32, hidden_dim: int = 128, num_types: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.intent_head = nn.Linear(hidden_dim, embed_dim)
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.title_head = nn.Linear(hidden_dim, embed_dim)

    def forward(self, profile: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(profile)
        return {
            "hypothesis": F.normalize(self.intent_head(hidden), dim=-1),
            "type_logits": self.type_head(hidden),
            "title_embedding": F.normalize(self.title_head(hidden), dim=-1),
        }


class CatalogueFulfilment(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, hypothesis: torch.Tensor, target_type: torch.Tensor, catalogue: torch.Tensor, catalogue_type: torch.Tensor, top_k: int = 24) -> Dict[str, torch.Tensor]:
        scores = hypothesis @ catalogue.T / self.temperature
        type_mask = catalogue_type.unsqueeze(0) == target_type.unsqueeze(1)
        scores = scores.masked_fill(~type_mask, -1e9)
        values, indices = torch.topk(scores, k=top_k, dim=1)
        return {"candidate_scores": values, "candidate_indices": indices}


class ShelfAligner(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.coherence = nn.Sequential(nn.Linear(embed_dim * 2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, hypothesis: torch.Tensor, candidate_embeddings: torch.Tensor, top_k: int = 8) -> Dict[str, torch.Tensor]:
        batch, candidates, dim = candidate_embeddings.shape
        expanded_hypothesis = hypothesis.unsqueeze(1).expand(batch, candidates, dim)
        pair_features = torch.cat([expanded_hypothesis, candidate_embeddings], dim=-1)
        alignment_score = self.coherence(pair_features).squeeze(-1)
        diversity_penalty = candidate_embeddings @ candidate_embeddings.transpose(1, 2)
        redundancy = diversity_penalty.mean(dim=-1)
        final_score = alignment_score - 0.05 * redundancy
        values, local_indices = torch.topk(final_score, k=top_k, dim=1)
        return {"aligned_scores": values, "local_indices": local_indices}


class HypothesisShelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = HypothesisPlanner()
        self.fulfilment = CatalogueFulfilment()
        self.aligner = ShelfAligner()

    def forward(self, profile: torch.Tensor, target_type: torch.Tensor, catalogue: torch.Tensor, catalogue_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        planned = self.planner(profile)
        fulfilled = self.fulfilment(planned["hypothesis"], target_type, catalogue, catalogue_type)
        candidate_embeddings = catalogue[fulfilled["candidate_indices"]]
        aligned = self.aligner(planned["hypothesis"], candidate_embeddings)
        final_indices = torch.gather(fulfilled["candidate_indices"], 1, aligned["local_indices"])
        return {**planned, **fulfilled, **aligned, "final_indices": final_indices}


def shelf_training_loss(outputs: Dict[str, torch.Tensor], target_type: torch.Tensor, positive_items: torch.Tensor) -> torch.Tensor:
    type_loss = F.cross_entropy(outputs["type_logits"], target_type)
    candidate_indices = outputs["candidate_indices"]
    labels = (candidate_indices.unsqueeze(-1) == positive_items.unsqueeze(1)).any(dim=-1).float()
    retrieval_loss = F.binary_cross_entropy_with_logits(outputs["candidate_scores"], labels)
    final_labels = (outputs["final_indices"].unsqueeze(-1) == positive_items.unsqueeze(1)).any(dim=-1).float()
    alignment_loss = F.binary_cross_entropy_with_logits(outputs["aligned_scores"], final_labels)
    return type_loss + retrieval_loss + alignment_loss
