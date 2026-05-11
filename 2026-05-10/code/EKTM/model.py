"""
EKTM: Effective Knowledge Transfer for Multi-Task Recommendation Models
arXiv: 2605.05730

Faithful PyTorch reproduction of the Router-Transmitter-Enhanced architecture
for multi-task CVR recommendation with cross-task knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class ExpertNetwork(nn.Module):
    """Shared expert MLP used as backbone for each task."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterModule(nn.Module):
    """
    Router module: integrates and distributes shared knowledge across tasks.

    The router aggregates representations from all tasks via attention-weighted
    pooling, producing a global cross-task knowledge vector.

    Formula (Section 3.2 of paper):
        r = SoftAttention([h_1, h_2, ..., h_K])  where K = num tasks
        r = sum_k alpha_k * h_k,   alpha = softmax(W_r * [h_1; ...; h_K])
    """

    def __init__(self, task_dim: int, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        # Attention scoring for cross-task aggregation
        self.attention = nn.Linear(task_dim * num_tasks, num_tasks)
        self.output_proj = nn.Linear(task_dim, task_dim)

    def forward(self, task_reprs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            task_reprs: list of K tensors, each (B, D)
        Returns:
            router_out: (B, D) cross-task knowledge vector
        """
        stacked = torch.cat(task_reprs, dim=-1)            # (B, K*D)
        weights = F.softmax(self.attention(stacked), dim=-1)   # (B, K)
        fused = torch.stack(task_reprs, dim=1)             # (B, K, D)
        # weighted sum across tasks
        router_out = (weights.unsqueeze(-1) * fused).sum(dim=1)  # (B, D)
        return self.output_proj(router_out)


class TransmitterModule(nn.Module):
    """
    Per-task Transmitter: transforms Router output into task-specific knowledge.

    Each CVR task gets its own transmitter to avoid cross-task interference.

    Formula:
        t_k = tanh(W_tk * r + b_tk)   (task-specific linear transform)
        knowledge_k = gate_k * t_k    (gating to control knowledge injection)
    """

    def __init__(self, task_dim: int):
        super().__init__()
        self.transform = nn.Linear(task_dim, task_dim)
        self.gate = nn.Sequential(
            nn.Linear(task_dim * 2, task_dim),
            nn.Sigmoid(),
        )

    def forward(self, router_out: torch.Tensor, task_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            router_out: (B, D) global knowledge from Router
            task_repr:  (B, D) current task representation
        Returns:
            transmitted: (B, D) adapted knowledge for this task
        """
        t = torch.tanh(self.transform(router_out))
        gate_input = torch.cat([router_out, task_repr], dim=-1)
        g = self.gate(gate_input)
        return g * t


class EnhancedModule(nn.Module):
    """
    Enhanced module: ensures transferred knowledge benefits the target task
    without corrupting original task signal.

    Implements residual integration with a learned scalar gating mechanism
    to control how much cross-task knowledge is injected.

    h_k_final = h_k_original + beta_k * transmitted_knowledge_k
    where beta_k is learned per-task scalar in [0, 1].
    """

    def __init__(self, task_dim: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))   # learned injection weight
        self.layer_norm = nn.LayerNorm(task_dim)

    def forward(self, task_repr: torch.Tensor, transmitted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_repr:   (B, D) original task representation
            transmitted: (B, D) knowledge from Transmitter
        Returns:
            enhanced: (B, D) final task representation
        """
        beta = torch.sigmoid(self.beta)
        enhanced = task_repr + beta * transmitted
        return self.layer_norm(enhanced)


class EKTMTaskTower(nn.Module):
    """Single CVR task tower with Transmitter + Enhanced modules."""

    def __init__(self, input_dim: int, hidden_dim: int, task_dim: int):
        super().__init__()
        self.task_expert = ExpertNetwork(input_dim, hidden_dim, task_dim)
        self.transmitter = TransmitterModule(task_dim)
        self.enhanced = EnhancedModule(task_dim)
        self.output_head = nn.Linear(task_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        router_out: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        task_repr = self.task_expert(x)

        if router_out is not None:
            transmitted = self.transmitter(router_out, task_repr)
            task_repr = self.enhanced(task_repr, transmitted)

        logit = self.output_head(task_repr)
        return {"logit": logit, "repr": task_repr}


class EKTM(nn.Module):
    """
    EKTM: Effective Knowledge Transfer for Multi-Task Recommendation Models

    Architecture:
        - Shared bottom: processes raw input features
        - Per-task experts: task-specific representation extraction
        - Router: aggregates cross-task knowledge
        - Per-task Transmitter: adapts router knowledge to each task
        - Enhanced module: residual integration with protection
        - Output head: binary CVR prediction per task

    Args:
        input_dim:    raw feature dimension (e.g., user+item embedding concat)
        shared_dim:   shared bottom output dimension
        hidden_dim:   expert hidden layer dimension
        task_dim:     task representation dimension
        num_tasks:    number of CVR tasks (e.g., 2 platform scenarios)
    """

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        hidden_dim: int,
        task_dim: int,
        num_tasks: int,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        # Shared bottom network
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim),
        )

        # Per-task towers
        self.task_towers = nn.ModuleList([
            EKTMTaskTower(shared_dim, hidden_dim, task_dim)
            for _ in range(num_tasks)
        ])

        # Router: aggregates across all tasks
        self.router = RouterModule(task_dim, num_tasks)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) raw user-item features
        Returns:
            dict with 'logits': (B, num_tasks) CVR prediction logits
                  and 'reprs':  (B, num_tasks, task_dim) task representations
        """
        shared = self.shared_bottom(x)

        # Pass 1: compute initial task representations (no cross-task transfer)
        init_reprs = []
        for tower in self.task_towers:
            out = tower(shared, router_out=None)
            init_reprs.append(out["repr"])

        # Router aggregates cross-task knowledge
        router_out = self.router(init_reprs)

        # Pass 2: apply Transmitter + Enhanced with router knowledge
        logits, reprs = [], []
        for tower in self.task_towers:
            out = tower(shared, router_out=router_out)
            logits.append(out["logit"])
            reprs.append(out["repr"])

        return {
            "logits": torch.cat(logits, dim=-1),    # (B, num_tasks)
            "reprs": torch.stack(reprs, dim=1),     # (B, num_tasks, task_dim)
        }
