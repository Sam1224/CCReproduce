import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(images), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 2048, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding(tokens)
        if mask is None:
            mask = tokens.ne(0).float()
        pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return F.normalize(self.proj(pooled), dim=-1)


class GradCIR(nn.Module):
    def __init__(self, vocab_size: int = 2048, embed_dim: int = 64):
        super().__init__()
        self.query_image = ImageEncoder(embed_dim)
        self.product_image = ImageEncoder(embed_dim)
        self.text = TextEncoder(vocab_size, embed_dim)
        self.query_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.product_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))

    def encode_query(self, image: torch.Tensor, modifier_tokens: torch.Tensor) -> torch.Tensor:
        visual = self.query_image(image)
        textual = self.text(modifier_tokens)
        return F.normalize(self.query_fusion(torch.cat([visual, textual], dim=-1)), dim=-1)

    def encode_product(self, image: torch.Tensor, title_tokens: torch.Tensor) -> torch.Tensor:
        visual = self.product_image(image)
        textual = self.text(title_tokens)
        return F.normalize(self.product_fusion(torch.cat([visual, textual], dim=-1)), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        query = self.encode_query(batch["query_image"], batch["modifier_tokens"])
        product = self.encode_product(batch["product_image"], batch["product_tokens"])
        return query @ product.T


def hierarchy_angular_loss(scores: torch.Tensor, relevance: torch.Tensor, margin: float = 0.08) -> torch.Tensor:
    target = relevance.float() / 3.0
    diagonal = scores.diag()
    regression = F.mse_loss((diagonal + 1.0) / 2.0, target)
    pair_delta = relevance[:, None] - relevance[None, :]
    score_delta = diagonal[:, None] - diagonal[None, :]
    rank_loss = F.relu(margin * pair_delta.sign().abs() - score_delta * pair_delta.sign())
    return regression + rank_loss[pair_delta > 0].mean().nan_to_num(0.0)


def ndcg_at_k(scores: torch.Tensor, relevance: torch.Tensor, k: int = 10) -> float:
    k = min(k, scores.numel())
    order = torch.argsort(scores, descending=True)[:k]
    gains = torch.pow(2.0, relevance[order].float()) - 1.0
    discounts = torch.log2(torch.arange(k, device=scores.device).float() + 2.0)
    dcg = (gains / discounts).sum()
    ideal = torch.sort(relevance.float(), descending=True).values[:k]
    idcg = ((torch.pow(2.0, ideal) - 1.0) / discounts).sum().clamp_min(1e-8)
    return float((dcg / idcg).detach().cpu())
