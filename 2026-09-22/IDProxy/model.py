import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalProxyEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embed_dim: int = 64):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, embed_dim),
        )
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        text = self.text_embedding(tokens).mean(dim=1)
        vision = self.image_encoder(image)
        return self.fusion(torch.cat([vision, text], dim=-1))


class CoarseToFineAligner(nn.Module):
    def __init__(self, embed_dim: int = 64, clusters: int = 32):
        super().__init__()
        self.cluster_head = nn.Linear(embed_dim, clusters)
        self.fine_mapper = nn.Sequential(nn.Linear(embed_dim + clusters, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, proxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_logits = self.cluster_head(proxy)
        coarse = F.gumbel_softmax(coarse_logits, tau=1.0, hard=False)
        aligned = self.fine_mapper(torch.cat([proxy, coarse], dim=-1))
        return F.normalize(aligned, dim=-1), coarse_logits


class IDProxyCTR(nn.Module):
    def __init__(self, users: int = 1024, items: int = 4096, vocab_size: int = 4096, embed_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(users, embed_dim)
        self.item_embedding = nn.Embedding(items, embed_dim)
        self.proxy_encoder = MultimodalProxyEncoder(vocab_size, embed_dim)
        self.aligner = CoarseToFineAligner(embed_dim)
        self.rank_head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user = self.user_embedding(batch["user_id"])
        id_item = self.item_embedding(batch["item_id"])
        proxy_raw = self.proxy_encoder(batch["image"], batch["tokens"])
        proxy, coarse_logits = self.aligner(proxy_raw)
        item = torch.where(batch["is_cold"].unsqueeze(-1).bool(), proxy, id_item)
        features = torch.cat([user, item, user * item, torch.abs(user - item)], dim=-1)
        logits = self.rank_head(features).squeeze(-1)
        return {"logits": logits, "proxy": proxy, "id_item": F.normalize(id_item, dim=-1), "coarse_logits": coarse_logits}


def idproxy_loss(output: dict[str, torch.Tensor], labels: torch.Tensor, is_cold: torch.Tensor, align_weight: float = 0.2) -> torch.Tensor:
    ctr_loss = F.binary_cross_entropy_with_logits(output["logits"], labels.float())
    align_loss = 1.0 - F.cosine_similarity(output["proxy"], output["id_item"].detach(), dim=-1)
    cold_weight = 0.5 + is_cold.float()
    return ctr_loss + align_weight * (align_loss * cold_weight).mean()
