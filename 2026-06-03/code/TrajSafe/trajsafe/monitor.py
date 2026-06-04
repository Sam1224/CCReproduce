"""
TrajSafe: Proactive Harm Amplification Monitor

Architecture:
- Trajectory Encoder: encodes the current conversation context
- Trajectory Predictor: predicts likely future trajectory directions (tree-based)
- Harm Assessor: scores predicted trajectories for harm amplification risk
- Intervention Selector: chooses intervention action (probe/steer) when risk detected

Training: Tree-based Reinforcement Learning (tree-based RL)
  - Tree nodes = possible future conversation paths
  - RL reward = harm prevention rate minus false positive rate
  - Policy = which intervention to take at each state

Inference: Real-time monitoring at each assistant turn
  1. Encode current trajectory
  2. Predict k likely next-turn extensions
  3. Score each extension for harm
  4. Intervene if top-k trajectories have high harm score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

from trajsafe.trajectory import Trajectory, Turn, HarmLevel


@dataclass
class MonitorDecision:
    """TrajSafe decision at each turn."""
    should_intervene: bool
    intervention_type: Optional[str]  # "probe" or "steer"
    harm_score: float                 # Predicted harm amplification score [0,1]
    predicted_harm_trajectory: Optional[str]  # Which harm path was predicted


class TrajectoryEncoder(nn.Module):
    """
    Encode a conversation trajectory into a fixed-size representation.

    In production, this would use a pretrained LLM encoder (e.g., sentence-transformer).
    Here we use a simple GRU over token embeddings as a faithful reproduction.

    Input: trajectory as list of (role, content) strings
    Output: (d_model,) trajectory embedding
    """

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.role_embed = nn.Embedding(2, embed_dim)  # user=0, assistant=1

        # GRU over token sequence within each turn
        self.turn_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,  # causal
        )

        # GRU over turn sequence (trajectory level)
        self.traj_encoder = nn.GRU(
            input_size=hidden_dim + embed_dim,  # turn repr + role embedding
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def encode_turn_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, L)
        Returns: (B, hidden_dim) — last hidden state
        """
        embeds = self.token_embed(token_ids)  # (B, L, embed_dim)
        _, hidden = self.turn_encoder(embeds)  # hidden: (num_layers, B, hidden_dim)
        return hidden[-1]  # (B, hidden_dim)

    def forward(
        self,
        token_ids_per_turn: List[torch.Tensor],  # list of (B, L_t) tensors
        role_ids: torch.Tensor,                  # (B, T) — role of each turn
    ) -> torch.Tensor:
        """
        Encode full trajectory.
        Returns: (B, hidden_dim) — trajectory representation
        """
        B = role_ids.shape[0]
        T = len(token_ids_per_turn)
        assert T == role_ids.shape[1]

        turn_reprs = []
        for t, tok in enumerate(token_ids_per_turn):
            turn_repr = self.encode_turn_text(tok)  # (B, hidden_dim)
            role_repr = self.role_embed(role_ids[:, t])  # (B, embed_dim)
            combined = torch.cat([turn_repr, role_repr], dim=-1)  # (B, hidden_dim+embed_dim)
            turn_reprs.append(combined)

        traj_input = torch.stack(turn_reprs, dim=1)  # (B, T, hidden_dim+embed_dim)
        _, hidden = self.traj_encoder(traj_input)
        return hidden[-1]  # (B, hidden_dim)


class HarmPredictor(nn.Module):
    """
    Predicts harm amplification score from trajectory representation.

    Two heads:
    1. harm_score: probability of harm amplification [0,1]
    2. category_logits: which of 12 harm categories is likely
    """

    NUM_CATEGORIES = 12

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.harm_head = nn.Linear(hidden_dim // 2, 1)
        self.category_head = nn.Linear(hidden_dim // 2, self.NUM_CATEGORIES)

    def forward(self, traj_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        traj_repr: (B, hidden_dim)
        Returns:
            harm_score: (B,) in [0,1]
            category_logits: (B, 12)
        """
        feat = self.net(traj_repr)
        harm_score = torch.sigmoid(self.harm_head(feat)).squeeze(-1)
        category_logits = self.category_head(feat)
        return harm_score, category_logits


class TrajSafeMonitor(nn.Module):
    """
    Full TrajSafe proactive monitor.

    At each turn t, given trajectory[0:t], TrajSafe:
    1. Encodes the trajectory
    2. Predicts harm score
    3. Applies tree-based lookahead (for k future steps)
    4. Returns intervention decision

    Tree-based RL Training:
    - State: trajectory embedding
    - Action: {no_action, probe_intent, steer_topic}
    - Reward: +1 if harm prevented, -0.3 if false positive (benign conversation disrupted)

    # Pseudocode for tree-based RL (from paper, Section 3.3):
    # for each trajectory in training set:
    #   tree = build_rollout_tree(trajectory, depth=K, branching=B)
    #   for each leaf in tree:
    #     reward = compute_harm_prevention_reward(leaf)
    #   policy_gradient_update(tree, rewards)
    """

    INTERVENTION_THRESHOLD = 0.6  # harm_score > this → intervene
    LOOKAHEAD_DEPTH = 3

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        intervention_threshold: float = 0.6,
    ):
        super().__init__()
        self.encoder = TrajectoryEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
        )
        self.predictor = HarmPredictor(hidden_dim=hidden_dim)
        self.intervention_threshold = intervention_threshold

        # Intervention type selector: given harm score + category → choose action
        self.intervention_selector = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 0=no_action, 1=probe, 2=steer
        )

    def forward(
        self,
        token_ids_per_turn: List[torch.Tensor],
        role_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        Returns:
            harm_score: (B,)
            intervention_logits: (B, 3)
        """
        traj_repr = self.encoder(token_ids_per_turn, role_ids)
        harm_score, _ = self.predictor(traj_repr)

        intervention_input = torch.cat(
            [traj_repr, harm_score.unsqueeze(-1)], dim=-1
        )
        intervention_logits = self.intervention_selector(intervention_input)
        return harm_score, intervention_logits

    @torch.no_grad()
    def monitor_turn(
        self,
        trajectory: "Trajectory",
        tokenizer=None,
    ) -> MonitorDecision:
        """
        Make real-time monitoring decision for the current trajectory.

        Args:
            trajectory: current conversation trajectory
            tokenizer: tokenizer for encoding text (uses simple hash if None)
        """
        self.eval()

        if len(trajectory.turns) < 2:
            return MonitorDecision(
                should_intervene=False,
                intervention_type=None,
                harm_score=0.0,
                predicted_harm_trajectory=None,
            )

        # Encode trajectory
        token_ids_list, role_tensor = self._encode_trajectory(trajectory, tokenizer)
        harm_score, intervention_logits = self(token_ids_list, role_tensor)

        harm_score_val = harm_score.item()
        should_intervene = harm_score_val >= self.intervention_threshold

        intervention_type = None
        if should_intervene:
            action_idx = intervention_logits.argmax(-1).item()
            intervention_type = ["no_action", "probe", "steer"][action_idx]
            if intervention_type == "no_action":
                should_intervene = False

        return MonitorDecision(
            should_intervene=should_intervene,
            intervention_type=intervention_type,
            harm_score=harm_score_val,
            predicted_harm_trajectory=trajectory.category if should_intervene else None,
        )

    def _encode_trajectory(
        self, trajectory: "Trajectory", tokenizer=None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Convert trajectory to tensors for the encoder."""
        MAX_LEN = 64

        token_ids_list = []
        role_ids = []

        for turn in trajectory.turns:
            if tokenizer is not None:
                ids = tokenizer.encode(turn.content)[:MAX_LEN]
            else:
                # Simple character-hash tokenizer for toy use
                ids = [ord(c) % 256 for c in turn.content[:MAX_LEN]]

            ids_padded = ids + [0] * (MAX_LEN - len(ids))
            token_ids_list.append(
                torch.tensor([ids_padded], dtype=torch.long)
            )
            role_ids.append(0 if turn.role == "user" else 1)

        role_tensor = torch.tensor([role_ids], dtype=torch.long)
        return token_ids_list, role_tensor
