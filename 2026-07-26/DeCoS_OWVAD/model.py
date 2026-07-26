import torch
import torch.nn as nn


class FrozenGateNet(nn.Module):
    def __init__(self, shared_direction: torch.Tensor, scale: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("shared_direction", shared_direction)
        self.scale = scale

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("btd,d->bt", visual_features, self.shared_direction)
        return torch.sigmoid(self.scale * logits)


class MarginComputer(nn.Module):
    def forward(
        self,
        visual_features: torch.Tensor,
        anomaly_embeddings: torch.Tensor,
        normal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normal_similarity = torch.einsum("btd,d->bt", visual_features, normal_embedding).unsqueeze(-1)
        anomaly_similarity = torch.einsum("btd,cd->btc", visual_features, anomaly_embeddings)
        margins = anomaly_similarity - normal_similarity
        delta_margins = margins - margins.mean(dim=1, keepdim=True)
        return margins, delta_margins


class DefinitionBlindScorer(nn.Module):
    def __init__(self, shared_direction: torch.Tensor) -> None:
        super().__init__()
        self.gate_net = FrozenGateNet(shared_direction)

    def forward(self, visual_features: torch.Tensor, anomaly_embeddings: torch.Tensor, normal_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, time_steps, num_classes = visual_features.size(0), visual_features.size(1), anomaly_embeddings.size(0)
        gate = self.gate_net(visual_features)
        scores = gate.unsqueeze(-1).expand(batch_size, time_steps, num_classes)
        zeros = torch.zeros_like(scores)
        return {"scores": scores, "gate": gate, "margins": zeros, "centered_margins": zeros, "residual": zeros}


class MarginOnlyScorer(nn.Module):
    def __init__(self, shared_direction: torch.Tensor) -> None:
        super().__init__()
        self.gate_net = FrozenGateNet(shared_direction)
        self.margin_computer = MarginComputer()

    def forward(self, visual_features: torch.Tensor, anomaly_embeddings: torch.Tensor, normal_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self.gate_net(visual_features)
        margins, _ = self.margin_computer(visual_features, anomaly_embeddings, normal_embedding)
        scores = gate.unsqueeze(-1) * margins
        zeros = torch.zeros_like(scores)
        return {"scores": scores, "gate": gate, "margins": margins, "centered_margins": zeros, "residual": zeros}


class CenteredMarginScorer(nn.Module):
    def __init__(self, shared_direction: torch.Tensor) -> None:
        super().__init__()
        self.gate_net = FrozenGateNet(shared_direction)
        self.margin_computer = MarginComputer()

    def forward(self, visual_features: torch.Tensor, anomaly_embeddings: torch.Tensor, normal_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self.gate_net(visual_features)
        margins, _ = self.margin_computer(visual_features, anomaly_embeddings, normal_embedding)
        centered_margins = margins - margins.mean(dim=-1, keepdim=True)
        scores = gate.unsqueeze(-1) * centered_margins
        residual = torch.zeros_like(scores)
        return {"scores": scores, "gate": gate, "margins": margins, "centered_margins": centered_margins, "residual": residual}


class DeCoSScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, shared_direction: torch.Tensor, rho: float = 0.6) -> None:
        super().__init__()
        self.gate_net = FrozenGateNet(shared_direction)
        self.margin_computer = MarginComputer()
        self.visual_projector = nn.Linear(feature_dim, hidden_dim)
        self.residual_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim + 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.residual_encoder[2].weight)
        nn.init.zeros_(self.residual_encoder[2].bias)
        self.num_classes = num_classes
        self.rho = rho

    def forward(self, visual_features: torch.Tensor, anomaly_embeddings: torch.Tensor, normal_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self.gate_net(visual_features)
        margins, delta_margins = self.margin_computer(visual_features, anomaly_embeddings, normal_embedding)
        centered_margins = margins - margins.mean(dim=-1, keepdim=True)

        visual_context = torch.relu(self.visual_projector(visual_features))
        visual_context = visual_context.unsqueeze(2).expand(-1, -1, self.num_classes, -1)
        margin_features = torch.stack([margins, delta_margins], dim=-1)
        joint_features = torch.cat([visual_context, margin_features], dim=-1)

        batch_size, time_steps, num_classes, channels = joint_features.shape
        residual_input = joint_features.permute(0, 2, 3, 1).reshape(batch_size * num_classes, channels, time_steps)
        residual_raw = self.residual_encoder(residual_input).reshape(batch_size, num_classes, time_steps).permute(0, 2, 1)
        residual = residual_raw - residual_raw.mean(dim=-1, keepdim=True)

        scores = gate.unsqueeze(-1) * (centered_margins + self.rho * residual)
        return {
            "scores": scores,
            "gate": gate,
            "margins": margins,
            "centered_margins": centered_margins,
            "residual": residual,
        }


if __name__ == "__main__":
    batch_size, time_steps, feature_dim, num_classes = 2, 64, 32, 4
    features = torch.randn(batch_size, time_steps, feature_dim)
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    anomalies = torch.randn(num_classes, feature_dim)
    anomalies = anomalies / anomalies.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.randn(feature_dim)
    normal = normal / normal.norm().clamp_min(1e-6)
    scorer = DeCoSScorer(feature_dim=feature_dim, hidden_dim=48, num_classes=num_classes, shared_direction=anomalies.mean(dim=0))
    result = scorer(features, anomalies, normal)
    print(result["scores"].shape)
