import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackAwareFlatQuantizer(nn.Module):
    def __init__(self, embed_dim: int = 64, codes: int = 128, commitment: float = 0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codes, embed_dim) * 0.02)
        self.commitment = commitment

    def forward(self, item: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(item, self.codebook)
        assignment = torch.argmin(distances, dim=-1)
        quantized = self.codebook[assignment]
        quantized_st = item + (quantized - item).detach()
        usage = torch.bincount(assignment, minlength=self.codebook.size(0)).float()
        usage = usage / usage.sum().clamp_min(1.0)
        balance_loss = (usage * torch.log(usage.clamp_min(1e-8) * self.codebook.size(0))).sum()
        commit_loss = F.mse_loss(item, quantized.detach()) + self.commitment * F.mse_loss(item.detach(), quantized)
        return quantized_st, assignment, commit_loss + 0.01 * balance_loss


class UNIQUE(nn.Module):
    def __init__(self, users: int = 2048, items: int = 8192, embed_dim: int = 64, codes: int = 128):
        super().__init__()
        self.user_embedding = nn.Embedding(users, embed_dim)
        self.item_features = nn.Embedding(items, embed_dim)
        self.quantizer = FeedbackAwareFlatQuantizer(embed_dim, codes)
        self.sequence_encoder = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.early_fusion = nn.Sequential(nn.Linear(embed_dim * 4, embed_dim * 2), nn.GELU(), nn.LayerNorm(embed_dim * 2))
        self.retrieval_head = nn.Linear(embed_dim * 2, codes)
        self.ctr_head = nn.Linear(embed_dim * 2, 1)
        self.duration_head = nn.Linear(embed_dim * 2, 1)
        self.completion_head = nn.Linear(embed_dim * 2, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user = self.user_embedding(batch["user_id"])
        history = self.item_features(batch["history_ids"])
        _, hidden = self.sequence_encoder(history)
        target = self.item_features(batch["target_id"])
        quantized, code, quant_loss = self.quantizer(target)
        fused = self.early_fusion(torch.cat([user, hidden.squeeze(0), quantized, user * quantized], dim=-1))
        return {
            "code_logits": self.retrieval_head(fused),
            "code": code,
            "ctr": self.ctr_head(fused).squeeze(-1),
            "duration": self.duration_head(fused).squeeze(-1),
            "completion": self.completion_head(fused).squeeze(-1),
            "quant_loss": quant_loss,
        }


def unique_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    retrieval = F.cross_entropy(output["code_logits"], output["code"].detach())
    ctr = F.binary_cross_entropy_with_logits(output["ctr"], batch["click"].float())
    completion = F.binary_cross_entropy_with_logits(output["completion"], batch["complete"].float())
    duration = F.mse_loss(torch.sigmoid(output["duration"]), batch["duration"].float())
    return retrieval + ctr + completion + duration + output["quant_loss"]
