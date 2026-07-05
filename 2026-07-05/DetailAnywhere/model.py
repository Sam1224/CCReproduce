import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class ViewBridgingTeacher(nn.Module):
    """Small DINO-style proxy teacher used for cross-modal feature alignment distillation."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(4, channels),
            nn.AvgPool2d(2),
            ConvBlock(channels, channels),
            nn.AvgPool2d(2),
            ConvBlock(channels, channels),
            nn.AdaptiveAvgPool2d(1),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(torch.cat([image, mask], dim=1)).flatten(1)
        return F.normalize(encoded, dim=-1)


class DetailAnywhere(nn.Module):
    """Toy DetailAnywhere implementation with focus conditioning, CFAD, and reward scoring."""

    def __init__(self, categories: int = 41, channels: int = 64, detail_size: int = 32):
        super().__init__()
        self.detail_size = detail_size
        self.category_embedding = nn.Embedding(categories, channels)
        self.reference_encoder = nn.Sequential(ConvBlock(4, channels), nn.AvgPool2d(2), ConvBlock(channels, channels))
        self.focus_encoder = nn.Sequential(ConvBlock(1, channels), nn.AvgPool2d(2), ConvBlock(channels, channels))
        self.alignment_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels))
        self.decoder = nn.Sequential(
            ConvBlock(channels * 2, channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(channels, channels // 2),
            nn.Conv2d(channels // 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self.reward_head = nn.Sequential(nn.Linear(channels + 3, channels), nn.SiLU(), nn.Linear(channels, 3))

    def forward(self, reference: torch.Tensor, focus_mask: torch.Tensor, category_id: torch.Tensor) -> dict:
        reference_features = self.reference_encoder(torch.cat([reference, focus_mask], dim=1))
        focus_features = self.focus_encoder(focus_mask)
        category_features = self.category_embedding(category_id).view(category_id.shape[0], -1, 1, 1)
        category_features = category_features.expand_as(reference_features)
        fused = torch.cat([reference_features + category_features, focus_features], dim=1)
        detail = self.decoder(fused)
        if detail.shape[-1] != self.detail_size:
            detail = F.interpolate(detail, size=(self.detail_size, self.detail_size), mode="bilinear", align_corners=False)
        aligned_reference = F.normalize(self.alignment_head(reference_features), dim=-1)
        aligned_detail = F.normalize(self.alignment_head(focus_features), dim=-1)
        pooled_color = detail.mean(dim=(2, 3))
        reward_logits = self.reward_head(torch.cat([aligned_reference, pooled_color], dim=1))
        return {
            "detail": detail,
            "aligned_reference": aligned_reference,
            "aligned_detail": aligned_detail,
            "reward_logits": reward_logits,
        }


def detailanywhere_loss(outputs: dict, target_detail: torch.Tensor, teacher_feature: torch.Tensor) -> dict:
    reconstruction = F.l1_loss(outputs["detail"], target_detail)
    cfad_reference = 1 - F.cosine_similarity(outputs["aligned_reference"], teacher_feature, dim=-1).mean()
    cfad_detail = 1 - F.cosine_similarity(outputs["aligned_detail"], teacher_feature, dim=-1).mean()
    identity = F.mse_loss(outputs["detail"].mean(dim=(2, 3)), target_detail.mean(dim=(2, 3)))
    target_reward = torch.ones_like(outputs["reward_logits"])
    reward = F.binary_cross_entropy_with_logits(outputs["reward_logits"], target_reward)
    total = reconstruction + 0.3 * (cfad_reference + cfad_detail) + 0.2 * identity + 0.1 * reward
    return {
        "total": total,
        "reconstruction": reconstruction.detach(),
        "cfad": (cfad_reference + cfad_detail).detach(),
        "identity": identity.detach(),
        "reward": reward.detach(),
    }
