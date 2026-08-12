from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TimeRoute(nn.Module):
    def __init__(
        self,
        num_items: int = 96,
        hidden_dim: int = 64,
        modal_dim: int = 32,
        max_length: int = 10,
        diffusion_steps: int = 3,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.text_projection = nn.Linear(modal_dim, hidden_dim)
        self.image_projection = nn.Linear(modal_dim, hidden_dim)
        self.audio_projection = nn.Linear(modal_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(max_length * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.graph_conditioner = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer("position_index", torch.arange(max_length), persistent=False)

    def encode_modalities(
        self,
        session: torch.Tensor,
        timestamps: torch.Tensor,
        catalog: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_tokens = self.text_projection(catalog["text_embeddings"][session])
        image_tokens = self.image_projection(catalog["image_embeddings"][session])
        audio_tokens = self.audio_projection(catalog["audio_embeddings"][session])
        recency = 1.0 - timestamps
        time_signal = torch.cat([timestamps, recency], dim=-1)
        time_state = self.time_mlp(time_signal)
        context = torch.cat(
            [
                text_tokens.mean(dim=1),
                image_tokens.mean(dim=1),
                audio_tokens.mean(dim=1),
                time_state,
            ],
            dim=-1,
        )
        route_logits = self.router(context)
        route_weights = route_logits.softmax(dim=-1)
        session_state = (
            route_weights[:, 0:1] * text_tokens.mean(dim=1)
            + route_weights[:, 1:2] * image_tokens.mean(dim=1)
            + route_weights[:, 2:3] * audio_tokens.mean(dim=1)
        )
        return session_state, route_weights, text_tokens, image_tokens + audio_tokens

    def denoise_items(
        self,
        route_weights: torch.Tensor,
        graph_edges: torch.Tensor,
        catalog: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        text_bank = self.text_projection(catalog["text_embeddings"])
        image_bank = self.image_projection(catalog["image_embeddings"])
        audio_bank = self.audio_projection(catalog["audio_embeddings"])
        fused_items = (
            route_weights[:, 0:1, None] * text_bank.unsqueeze(0)
            + route_weights[:, 1:2, None] * image_bank.unsqueeze(0)
            + route_weights[:, 2:3, None] * audio_bank.unsqueeze(0)
        )
        graph_context = torch.matmul(graph_edges.unsqueeze(0), fused_items)
        condition = torch.cat([graph_context, route_weights.unsqueeze(1).expand(-1, graph_context.size(1), -1)], dim=-1)
        condition = self.graph_conditioner(condition)
        noisy = fused_items + 0.1 * torch.randn_like(fused_items)
        denoised = noisy
        for _ in range(self.diffusion_steps):
            denoised = self.denoiser(torch.cat([denoised, condition], dim=-1))
        return denoised

    def forward(self, session: torch.Tensor, timestamps: torch.Tensor, catalog: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        session_state, route_weights, _, _ = self.encode_modalities(session, timestamps, catalog)
        denoised_items = self.denoise_items(route_weights, catalog["graph_edges"], catalog)
        query = self.query_head(session_state)
        logits = torch.einsum("bd,bid->bi", query, denoised_items)
        return {
            "logits": logits,
            "route_weights": route_weights,
            "denoised_items": denoised_items,
        }

    def loss(
        self,
        session: torch.Tensor,
        timestamps: torch.Tensor,
        target: torch.Tensor,
        catalog: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.forward(session, timestamps, catalog)
        ranking_loss = F.cross_entropy(outputs["logits"], target)
        text_bank = self.text_projection(catalog["text_embeddings"])
        image_bank = self.image_projection(catalog["image_embeddings"])
        audio_bank = self.audio_projection(catalog["audio_embeddings"])
        target_fused = (
            outputs["route_weights"][:, 0:1, None] * text_bank.unsqueeze(0)
            + outputs["route_weights"][:, 1:2, None] * image_bank.unsqueeze(0)
            + outputs["route_weights"][:, 2:3, None] * audio_bank.unsqueeze(0)
        )
        diffusion_loss = F.mse_loss(outputs["denoised_items"], target_fused)
        entropy = -(outputs["route_weights"] * torch.log(outputs["route_weights"] + 1e-8)).sum(dim=-1).mean()
        total_loss = ranking_loss + 0.15 * diffusion_loss + 0.02 * entropy
        predictions = outputs["logits"].argmax(dim=-1)
        metrics = {
            "top1": (predictions == target).float().mean().item(),
            "ranking_loss": ranking_loss.item(),
            "diffusion_loss": diffusion_loss.item(),
            "router_entropy": entropy.item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def evaluate(
        self,
        session: torch.Tensor,
        timestamps: torch.Tensor,
        target: torch.Tensor,
        catalog: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        logits = self.forward(session, timestamps, catalog)["logits"]
        top1 = logits.argmax(dim=-1)
        top5 = logits.topk(k=5, dim=-1).indices
        top10 = logits.topk(k=10, dim=-1).indices
        target_expanded = target.unsqueeze(1)
        matched = top10 == target_expanded
        rank_positions = matched.float().argmax(dim=1) + 1
        ndcg10 = torch.where(
            matched.any(dim=1),
            1.0 / torch.log2(rank_positions.float() + 1.0),
            torch.zeros_like(rank_positions, dtype=torch.float32),
        )
        return {
            "recall@1": (top1 == target).float().mean().item(),
            "recall@5": (top5 == target_expanded).any(dim=1).float().mean().item(),
            "recall@10": (top10 == target_expanded).any(dim=1).float().mean().item(),
            "ndcg@10": ndcg10.mean().item(),
        }
