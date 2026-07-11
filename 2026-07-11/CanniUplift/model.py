import torch
import torch.nn as nn
import torch.nn.functional as F


class TreatAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim + 2, hidden_dim)
        self.value = nn.Linear(feature_dim + 2, hidden_dim)

    def forward(self, user: torch.Tensor, seller_features: torch.Tensor, treatment: torch.Tensor, coupon_value: torch.Tensor) -> torch.Tensor:
        treatment_features = torch.stack([treatment, coupon_value], dim=-1)
        seller_treatment = torch.cat([seller_features, treatment_features], dim=-1)
        query = self.query(user).unsqueeze(1)
        key = self.key(seller_treatment)
        value = self.value(seller_treatment)
        weights = torch.softmax((query * key).sum(-1) / key.shape[-1] ** 0.5, dim=1)
        return value * weights.unsqueeze(-1)


class CanniUplift(nn.Module):
    """Toy CanniUplift with PGA, RDD, and treatment attention."""

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.treat_attention = TreatAttention(feature_dim, hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(feature_dim * 2 + hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.control_head = nn.Linear(hidden_dim, 1)
        self.redeem_head = nn.Linear(hidden_dim, 1)
        self.non_redeem_head = nn.Linear(hidden_dim, 1)
        self.platform_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, user: torch.Tensor, seller_features: torch.Tensor, treatment: torch.Tensor, coupon_value: torch.Tensor) -> dict:
        user_expanded = user.unsqueeze(1).expand_as(seller_features)
        attended = self.treat_attention(user, seller_features, treatment, coupon_value)
        inputs = torch.cat([user_expanded, seller_features, attended, treatment.unsqueeze(-1), coupon_value.unsqueeze(-1)], dim=-1)
        hidden = self.shared(inputs)
        control = self.control_head(hidden).squeeze(-1)
        redeem_path = self.redeem_head(hidden).squeeze(-1)
        non_redeem_path = self.non_redeem_head(hidden).squeeze(-1)
        predicted_redeem = torch.sigmoid(redeem_path)
        predicted_uplift = treatment * (redeem_path - control) + treatment * (1.0 - predicted_redeem.detach()) * (non_redeem_path - control)
        platform_delta = self.platform_head(hidden.mean(dim=1))
        return {
            "control": control,
            "redeem_path": redeem_path,
            "non_redeem_path": non_redeem_path,
            "redemption_prob": predicted_redeem,
            "seller_uplift": predicted_uplift,
            "platform_delta": platform_delta,
        }


def canniuplift_loss(outputs: dict, batch: dict, pga_weight: float = 0.5, rdd_weight: float = 0.2) -> dict:
    seller_loss = F.mse_loss(outputs["seller_uplift"], batch["seller_uplift"])
    redemption_loss = F.binary_cross_entropy(outputs["redemption_prob"].clamp(1e-5, 1 - 1e-5), batch["redemption"])
    pga_loss = F.mse_loss(outputs["platform_delta"], batch["platform_delta"])
    denoise_target = batch["seller_uplift"] * batch["redemption"]
    rdd_loss = F.mse_loss(outputs["redeem_path"] - outputs["non_redeem_path"], denoise_target)
    total = seller_loss + pga_weight * pga_loss + rdd_weight * (redemption_loss + rdd_loss)
    return {"total": total, "seller": seller_loss.detach(), "pga": pga_loss.detach(), "rdd": rdd_loss.detach()}
