from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def grad_reverse(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(tensor, scale)


class SourceAwareRanker(nn.Module):
    def __init__(self, num_sources: int, source_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.source_embedding = nn.Embedding(num_sources, source_dim)
        item_dim = 3 + source_dim
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.source_probe = nn.Linear(hidden_dim, num_sources)

    def encode_item(self, features: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        source_vector = self.source_embedding(source)
        return self.item_encoder(torch.cat([features, source_vector], dim=-1))

    def score_item(self, features: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.utility_head(self.encode_item(features, source)).squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor], grl_scale: float = 0.0) -> Dict[str, torch.Tensor]:
        left_repr = self.encode_item(batch["left_features"], batch["left_source"])
        right_repr = self.encode_item(batch["right_features"], batch["right_source"])
        left_score = self.utility_head(left_repr).squeeze(-1)
        right_score = self.utility_head(right_repr).squeeze(-1)
        pair_logit = left_score - right_score
        left_probe = self.source_probe(grad_reverse(left_repr, grl_scale))
        right_probe = self.source_probe(grad_reverse(right_repr, grl_scale))
        return {
            "pair_logit": pair_logit,
            "left_score": left_score,
            "right_score": right_score,
            "left_probe": left_probe,
            "right_probe": right_probe,
        }


def debiasing_loss(
    model: SourceAwareRanker,
    batch: Dict[str, torch.Tensor],
    cf_batch: Dict[str, torch.Tensor],
    adversarial_weight: float = 0.2,
    consistency_weight: float = 2.0,
    grl_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    output = model(batch, grl_scale=grl_scale)
    cf_output = model(cf_batch, grl_scale=0.0)
    ranking_loss = F.binary_cross_entropy_with_logits(output["pair_logit"], batch["label"])
    source_loss = 0.5 * (
        F.cross_entropy(output["left_probe"], batch["left_source"])
        + F.cross_entropy(output["right_probe"], batch["right_source"])
    )
    consistency_loss = F.mse_loss(output["pair_logit"], cf_output["pair_logit"])
    total = ranking_loss + adversarial_weight * source_loss + consistency_weight * consistency_loss
    return total, {
        "ranking_loss": float(ranking_loss.detach()),
        "source_loss": float(source_loss.detach()),
        "consistency_loss": float(consistency_loss.detach()),
    }
