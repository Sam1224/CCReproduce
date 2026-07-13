import torch
import torch.nn as nn


NUM_ACTIONS = 4


class DifficultyRouter(nn.Module):
    """Lightweight router that predicts whether a session needs escalated control."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, conversation_vec: torch.Tensor) -> torch.Tensor:
        return self.net(conversation_vec)


class WritePlanVerifier(nn.Module):
    """Scores whether a candidate backend write is consistent with the session constraints."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def action_one_hot(action_id: torch.Tensor, num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    return torch.nn.functional.one_hot(action_id, num_classes=num_actions).float()


def build_verifier_features(conversation_vec: torch.Tensor, slot_vec: torch.Tensor, action_id: torch.Tensor) -> torch.Tensor:
    overlap = torch.minimum(conversation_vec, slot_vec).sum(dim=-1, keepdim=True)
    slot_mass = slot_vec.sum(dim=-1, keepdim=True)
    return torch.cat([conversation_vec, slot_vec, action_one_hot(action_id), overlap, slot_mass], dim=-1)


def difficulty_routed_control(batch: dict, router: DifficultyRouter, verifier: WritePlanVerifier, threshold: float = 0.55) -> dict:
    router_logits = router(batch["conversation_vec"])
    complex_prob = torch.softmax(router_logits, dim=-1)[:, 1]
    escalated = complex_prob >= threshold

    slot_scores = []
    for slot_index in range(batch["slot_vectors"].shape[1]):
        features = build_verifier_features(batch["conversation_vec"], batch["slot_vectors"][:, slot_index, :], batch["action_id"])
        slot_scores.append(verifier(features).squeeze(-1))
    verifier_scores = torch.stack(slot_scores, dim=1)
    verifier_scores = verifier_scores.masked_fill(batch["slot_mask"] < 0.5, -1e9)
    verifier_choice = verifier_scores.argmax(dim=1)

    baseline_choice = (batch["slot_mask"].sum(dim=1).long() - 1).clamp(min=0)
    chosen_slot = torch.where(escalated, verifier_choice, baseline_choice)

    return {
        "router_logits": router_logits,
        "complex_prob": complex_prob,
        "escalated": escalated,
        "verifier_scores": verifier_scores,
        "baseline_choice": baseline_choice,
        "chosen_slot": chosen_slot,
    }
