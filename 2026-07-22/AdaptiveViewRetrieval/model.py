from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PerceptualViewBank(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        original = images
        centered = (images - images.mean(dim=(-2, -1), keepdim=True)) * 1.8 + 0.5
        contrast = centered.clamp(0, 1)
        lowpass = F.avg_pool2d(images, kernel_size=7, stride=1, padding=3)
        gray = images.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6).repeat(1, 3, 1, 1)
        edges = edges / edges.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        inverted = 1.0 - images
        return torch.stack([original, contrast, lowpass, edges, inverted], dim=1)


class FrozenFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.net(images), dim=-1)


class AdaptiveViewRetrieval(nn.Module):
    def __init__(self, embedding_dim: int = 128, projected_dim: int = 96, num_views: int = 5):
        super().__init__()
        self.view_bank = PerceptualViewBank()
        self.encoder = FrozenFeatureEncoder(embedding_dim)
        self.num_views = num_views
        self.projections = nn.ModuleList([nn.Linear(embedding_dim, projected_dim) for _ in range(num_views)])
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * num_views, 192),
            nn.GELU(),
            nn.Linear(192, num_views),
        )
        self.calibrator = nn.Sequential(
            nn.Linear(projected_dim, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(1.5))

    def encode_views(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        views = self.view_bank(images)
        batch, num_views, channels, height, width = views.shape
        features = self.encoder(views.view(batch * num_views, channels, height, width)).view(batch, num_views, -1)
        projected = []
        for view_index, projection in enumerate(self.projections):
            projected.append(F.normalize(projection(features[:, view_index]), dim=-1))
        return features, torch.stack(projected, dim=1)

    def forward(self, images: torch.Tensor, templates: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_features, image_projected = self.encode_views(images)
        with torch.no_grad():
            _, template_projected = self.encode_views(templates)
        gate_logits = self.gate(image_features.flatten(1))
        gate_weights = F.softmax(gate_logits, dim=-1)
        view_scores = torch.einsum("bvd,mvd->bvm", image_projected, template_projected)
        retrieval_logits = (gate_weights.unsqueeze(-1) * view_scores).sum(dim=1) * self.log_temperature.exp()
        fused = (gate_weights.unsqueeze(-1) * image_projected).sum(dim=1)
        harmful_logit = self.calibrator(fused).squeeze(-1)
        return {
            "retrieval_logits": retrieval_logits,
            "harmful_logit": harmful_logit,
            "gate_weights": gate_weights,
            "view_scores": view_scores,
        }


def avr_loss(outputs: Dict[str, torch.Tensor], message_id: torch.Tensor, harmful: torch.Tensor, gate_entropy_weight: float = 0.01) -> Tuple[torch.Tensor, Dict[str, float]]:
    retrieval_loss = F.cross_entropy(outputs["retrieval_logits"], message_id)
    calibration_loss = F.binary_cross_entropy_with_logits(outputs["harmful_logit"], harmful)
    gate = outputs["gate_weights"].clamp_min(1e-8)
    entropy = -(gate * gate.log()).sum(dim=-1).mean()
    loss = retrieval_loss + 0.5 * calibration_loss - gate_entropy_weight * entropy
    with torch.no_grad():
        retrieval_acc = (outputs["retrieval_logits"].argmax(dim=-1) == message_id).float().mean().item()
        harmful_pred = (outputs["harmful_logit"].sigmoid() >= 0.5).float()
        moderation_acc = (harmful_pred == harmful).float().mean().item()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "retrieval_loss": float(retrieval_loss.detach().cpu()),
        "calibration_loss": float(calibration_loss.detach().cpu()),
        "gate_entropy": float(entropy.detach().cpu()),
        "retrieval_acc": retrieval_acc,
        "moderation_acc": moderation_acc,
    }
