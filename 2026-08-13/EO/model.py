from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from data import BELIEF_DIM, FRICTION_INDEX, NUM_ARMS, OBS_DIM, RESISTANCE_INDEX, STEP_INPUT_DIM


class BeliefTracker(nn.Module):
    def __init__(self, input_dim: int = STEP_INPUT_DIM, hidden_dim: int = 64, belief_dim: int = BELIEF_DIM) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.belief_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, belief_dim))

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.gru(sequence)
        index = (lengths - 1).clamp_min(0)
        hidden = encoded[torch.arange(sequence.size(0), device=sequence.device), index]
        belief = torch.sigmoid(self.belief_head(hidden))
        return hidden, belief


class ExperienceOrchestrator(nn.Module):
    def __init__(self, input_dim: int = STEP_INPUT_DIM, hidden_dim: int = 64, num_arms: int = NUM_ARMS) -> None:
        super().__init__()
        self.belief_tracker = BeliefTracker(input_dim=input_dim, hidden_dim=hidden_dim)
        self.bandit = nn.Sequential(
            nn.Linear(hidden_dim + BELIEF_DIM + OBS_DIM + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_arms),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + BELIEF_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.kp = 0.90
        self.ki = 0.35
        self.kd = 0.20

    def _gather_latest(self, tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        index = (lengths - 1).clamp_min(0)
        return tensor[torch.arange(tensor.size(0), device=tensor.device), index]

    def _pid_control(self, sequence: torch.Tensor, lengths: torch.Tensor, belief: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        resistances = sequence[:, :, RESISTANCE_INDEX]
        current_resistance = self._gather_latest(resistances, lengths)
        previous_index = (lengths - 2).clamp_min(0)
        previous_resistance = resistances[torch.arange(sequence.size(0), device=sequence.device), previous_index]

        positions = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0)
        mask = (positions < lengths.unsqueeze(1)).float()
        target_resistance = 0.32 + 0.28 * (1.0 - belief[:, 0]) + 0.18 * belief[:, 2]
        centered = (resistances - target_resistance.unsqueeze(1)) * mask
        integral = centered.sum(dim=1) / lengths.float().clamp_min(1.0)
        derivative = current_resistance - previous_resistance
        error = current_resistance - target_resistance
        control = self.kp * error + self.ki * integral + self.kd * derivative
        return control, target_resistance

    def _pid_bias(self, control: torch.Tensor, belief: torch.Tensor, latest_obs: torch.Tensor) -> torch.Tensor:
        positive = F.relu(control)
        negative = F.relu(-control)
        readiness = belief[:, 0]
        info_need = belief[:, 1]
        trust_need = belief[:, 2]
        friction = latest_obs[:, FRICTION_INDEX]
        return torch.stack(
            [
                0.30 * positive + 0.20 * info_need + 0.05 * friction,
                0.55 * positive + 0.20 * trust_need,
                0.10 + 0.10 * readiness + 0.05 * negative,
                -0.90 * positive + 0.75 * negative + 0.45 * readiness - 0.25 * info_need - 0.15 * friction,
            ],
            dim=1,
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden, belief = self.belief_tracker(sequence, lengths)
        latest_obs = self._gather_latest(sequence[:, :, :OBS_DIM], lengths)
        control, target_resistance = self._pid_control(sequence, lengths, belief)
        bandit_input = torch.cat([hidden, belief, latest_obs, control.unsqueeze(1)], dim=1)
        logits = self.bandit(bandit_input) + self._pid_bias(control, belief, latest_obs)
        value = self.value_head(torch.cat([hidden, belief], dim=1)).squeeze(-1)
        return {
            "logits": logits,
            "belief": belief,
            "control": control,
            "value": value,
            "latest_obs": latest_obs,
            "target_resistance": target_resistance,
        }

    def loss(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        target_arm: torch.Tensor,
        belief_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.forward(sequence, lengths)
        action_loss = F.cross_entropy(outputs["logits"], target_arm)
        belief_loss = F.mse_loss(outputs["belief"], belief_target)
        value_target = belief_target[:, 0] - outputs["latest_obs"][:, RESISTANCE_INDEX] - 0.40 * outputs["latest_obs"][:, FRICTION_INDEX]
        value_loss = F.mse_loss(outputs["value"], value_target)
        loss = action_loss + 0.50 * belief_loss + 0.20 * value_loss
        metrics = {
            "action_acc": (outputs["logits"].argmax(dim=-1) == target_arm).float().mean().item(),
            "belief_mae": (outputs["belief"] - belief_target).abs().mean().item(),
            "avg_control": outputs["control"].mean().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def act(self, sequence: torch.Tensor, length: int):
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        lengths = torch.tensor([length], dtype=torch.long, device=sequence.device)
        outputs = self.forward(sequence, lengths)
        action = int(outputs["logits"].argmax(dim=-1).item())
        diagnostics = {
            "belief": outputs["belief"].squeeze(0).detach().cpu().tolist(),
            "control": float(outputs["control"].item()),
            "target_resistance": float(outputs["target_resistance"].item()),
        }
        return action, diagnostics


class NaiveBaseline:
    @torch.no_grad()
    def act(self, sequence: torch.Tensor, length: int):
        latest = sequence[length - 1, :OBS_DIM].detach().cpu()
        readiness_signal = float(latest[0].item())
        info_signal = float(latest[1].item())
        trust_signal = float(latest[2].item())
        resistance = float(latest[RESISTANCE_INDEX].item())
        friction = float(latest[FRICTION_INDEX].item())
        stage = float(latest[5].item())

        if resistance > 0.72:
            return 1, {"policy": "empathize-high-resistance"}
        if stage < 0.40 and (info_signal > 0.64 or friction > 0.60):
            return 0, {"policy": "educate-early"}
        if readiness_signal > 0.58 and resistance < 0.42 and stage > 0.35:
            return 3, {"policy": "early-cta"}
        if trust_signal > 0.55:
            return 2, {"policy": "social-proof"}
        return 1, {"policy": "empathetic-default"}
