"""
OCL PolicyLayer

Implements the two-stage policy enforcement at the execution boundary:
  1. Symbolic rule check (fast, hard constraints)
  2. MLP classifier (learned soft-constraint detector)

Decision outcomes:
  APPROVE  — action is safe, pass to environment
  BLOCK    — action clearly violates policy, reject immediately
  ESCALATE — action is ambiguous, route to EscalationHandler
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import List

from data import (ACTION_TYPES, AUTHORIZED_DISCOUNT_MAX, PRICE_DELTA_MAX,
                  AUTHORIZED_CONTRACT_FIELDS, action_to_feature_vector)


class Decision(Enum):
    APPROVE = "APPROVE"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


@dataclass
class PolicyDecision:
    decision: Decision
    confidence: float   # 0-1
    reason: str


FEATURE_DIM = 12  # matches action_to_feature_vector output


class PolicyMLP(nn.Module):
    """Small MLP that outputs P(violation | action features)."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SymbolicRules:
    """Hard, interpretable rules — checked before the MLP."""

    @staticmethod
    def check(action: dict) -> PolicyDecision | None:
        """Return a decision if a hard rule fires, else None (fall through to MLP)."""
        atype = action.get("type")

        if atype == "price_update":
            delta = abs(action.get("delta_pct", 0.0))
            if delta > 0.9:  # extreme price manipulation → hard block
                return PolicyDecision(Decision.BLOCK, 1.0, "price delta > 90% (hard rule)")

        elif atype == "discount_apply":
            d = action.get("discount_pct", 0.0)
            if d > 0.55:  # clearly above any authorized ceiling
                return PolicyDecision(Decision.BLOCK, 1.0, f"discount {d:.0%} exceeds hard ceiling")

        elif atype == "contract_modify":
            fields = set(action.get("fields", []))
            unauthorized = fields - AUTHORIZED_CONTRACT_FIELDS
            if "liability" in unauthorized or "arbitration" in unauthorized:
                return PolicyDecision(Decision.BLOCK, 1.0, f"unauthorized contract fields: {unauthorized}")

        elif atype == "message_send":
            if action.get("contains_threat", False):
                return PolicyDecision(Decision.BLOCK, 1.0, "threatening message (hard rule)")

        return None  # no hard rule fired


class PolicyLayer:
    """
    Combines symbolic rules with a learned MLP classifier.

    approve_threshold: P(violation) below this → APPROVE
    block_threshold:   P(violation) above this → BLOCK
    else               ESCALATE (ambiguous)
    """

    def __init__(self, model_path: str | None = None,
                 approve_threshold: float = 0.2,
                 block_threshold: float = 0.7):
        self.approve_threshold = approve_threshold
        self.block_threshold = block_threshold
        self.rules = SymbolicRules()

        self.mlp = PolicyMLP()
        if model_path:
            self.mlp.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.mlp.eval()

    @torch.no_grad()
    def enforce(self, action: dict) -> PolicyDecision:
        # Stage 1: symbolic hard rules
        hard = self.rules.check(action)
        if hard is not None:
            return hard

        # Stage 2: MLP soft classifier
        feat = action_to_feature_vector(action)
        x = torch.tensor([feat], dtype=torch.float32)
        p_violation = self.mlp(x).item()

        if p_violation < self.approve_threshold:
            return PolicyDecision(Decision.APPROVE, 1 - p_violation,
                                  f"MLP P(viol)={p_violation:.3f} < approve_thresh")
        elif p_violation > self.block_threshold:
            return PolicyDecision(Decision.BLOCK, p_violation,
                                  f"MLP P(viol)={p_violation:.3f} > block_thresh")
        else:
            return PolicyDecision(Decision.ESCALATE, p_violation,
                                  f"MLP P(viol)={p_violation:.3f} in escalation zone")


# ---------------------------------------------------------------------------
# Helper: load policy from checkpoint path (or create untrained)
# ---------------------------------------------------------------------------

def load_policy(ckpt: str | None = None, **kwargs) -> PolicyLayer:
    return PolicyLayer(model_path=ckpt, **kwargs)
