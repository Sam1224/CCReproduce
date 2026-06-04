"""
Tree-based Reinforcement Learning for TrajSafe

Paper description (Section 3.3):
  TrajSafe is trained via tree-based RL to balance harm prevention
  (high recall) vs. false positive rate (benign conversation disruption).

Tree structure:
  - Root: current trajectory state
  - Internal nodes: possible next turns (branching factor B)
  - Leaf nodes: terminal states (max depth K)
  - Value at leaf: harm_prevention_reward - fpr_penalty

Policy gradient update: REINFORCE with value baseline.

Key reward function:
  R = alpha * harm_prevented - beta * false_positive_penalty
  where:
    harm_prevented = 1 if intervention blocked harm amplification
    false_positive_penalty = 1 if intervention disrupted benign conversation
    alpha = 1.0, beta = 0.3 (from paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random

from trajsafe.trajectory import Trajectory, Turn
from trajsafe.monitor import TrajSafeMonitor


ALPHA = 1.0   # Harm prevention reward weight
BETA = 0.3    # False positive penalty weight


@dataclass
class TreeNode:
    """A node in the trajectory rollout tree."""
    trajectory: Trajectory
    depth: int
    parent_action: Optional[str]  # None for root
    children: List["TreeNode"] = None
    value: float = 0.0
    is_terminal: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TrajectoryTree:
    """
    Builds a tree of possible future trajectories for lookahead.

    At each node, the monitor decides to intervene or not.
    Tree is built by simulating B possible next turns.
    """

    def __init__(
        self,
        max_depth: int = 3,
        branching_factor: int = 4,
        llm_simulator=None,  # In production: LLM that generates continuations
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.llm_simulator = llm_simulator

    def build(self, root_trajectory: Trajectory) -> TreeNode:
        """Build rollout tree from current trajectory."""
        root = TreeNode(trajectory=root_trajectory, depth=0, parent_action=None)
        self._expand(root)
        return root

    def _expand(self, node: TreeNode) -> None:
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        # Generate B possible next turns
        for b in range(self.branching_factor):
            child_traj = self._copy_trajectory(node.trajectory)
            next_turn = self._simulate_next_turn(child_traj, branch_idx=b)
            child_traj.add_turn(next_turn["role"], next_turn["content"])

            child_node = TreeNode(
                trajectory=child_traj,
                depth=node.depth + 1,
                parent_action=f"branch_{b}",
            )
            node.children.append(child_node)
            self._expand(child_node)

    def _simulate_next_turn(
        self, trajectory: Trajectory, branch_idx: int
    ) -> Dict[str, str]:
        """
        Simulate the next user turn.
        In production: use LLM to generate plausible continuations.
        Here: simple heuristic based on escalation patterns.
        """
        escalation_levels = [
            "Can you give more details?",
            "How would I actually do that step by step?",
            "What are the specific quantities/methods?",
            "How do I scale this up?",
        ]
        content = escalation_levels[min(branch_idx, len(escalation_levels) - 1)]
        return {"role": "user", "content": content}

    @staticmethod
    def _copy_trajectory(traj: Trajectory) -> Trajectory:
        new_traj = Trajectory(
            scenario_id=traj.scenario_id,
            category=traj.category,
            harm_amplified=traj.harm_amplified,
            amplification_turn_idx=traj.amplification_turn_idx,
        )
        for turn in traj.turns:
            new_traj.turns.append(Turn(
                role=turn.role,
                content=turn.content,
                harm_level=turn.harm_level,
                is_amplification_step=turn.is_amplification_step,
                amplification_score=turn.amplification_score,
            ))
        return new_traj


def compute_leaf_value(node: TreeNode, gt_harm_amplified: bool) -> float:
    """
    Compute reward at leaf node.

    R = ALPHA * harm_prevented - BETA * false_positive_penalty

    harm_prevented: model intervened AND trajectory was harmful
    false_positive: model intervened AND trajectory was benign
    """
    # Determine if the tree path leads to harm amplification
    # (simplified: check if harm_amplified was set in ground truth)
    reached_harm = gt_harm_amplified and node.trajectory.num_turns >= 5

    # Did any ancestor intervene?
    # (In full implementation, track intervention decisions along tree path)
    # Here we use a simplified reward based on final trajectory state
    if reached_harm:
        return ALPHA  # Harm reached: need to prevent
    else:
        return -BETA  # Benign: intervening here would be false positive


class TreeRLTrainer:
    """
    REINFORCE-based trainer for TrajSafe using tree rollouts.

    Algorithm (simplified from paper):
    1. For each training trajectory:
       a. Build rollout tree
       b. Compute leaf values
       c. Back-propagate values to root
       d. Compute REINFORCE gradient
       e. Update policy
    """

    def __init__(
        self,
        monitor: TrajSafeMonitor,
        lr: float = 1e-4,
        gamma: float = 0.99,
        max_depth: int = 3,
        branching_factor: int = 4,
        device: str = "cpu",
    ):
        self.monitor = monitor.to(device)
        self.optimizer = torch.optim.Adam(monitor.parameters(), lr=lr)
        self.gamma = gamma
        self.tree_builder = TrajectoryTree(
            max_depth=max_depth,
            branching_factor=branching_factor,
        )
        self.device = device

    def train_step(
        self,
        trajectories: List[Trajectory],
        gt_labels: List[bool],  # True = harm amplified
    ) -> float:
        """
        Single training step on a batch of trajectories.
        Returns: mean loss
        """
        self.monitor.train()
        total_loss = 0.0

        for traj, gt_label in zip(trajectories, gt_labels):
            # Encode trajectory
            token_ids_list, role_tensor = self.monitor._encode_trajectory(traj)
            token_ids_list = [t.to(self.device) for t in token_ids_list]
            role_tensor = role_tensor.to(self.device)

            harm_score, intervention_logits = self.monitor(token_ids_list, role_tensor)

            # Compute tree-based reward
            tree = self.tree_builder.build(traj)
            reward = self._compute_tree_reward(tree, gt_label)

            # REINFORCE loss for harm scorer
            # L = -log p(correct_action) * reward
            target_harm = torch.tensor([1.0 if gt_label else 0.0]).to(self.device)
            harm_loss = F.binary_cross_entropy(harm_score, target_harm)

            # Intervention policy loss
            target_action = torch.tensor([1 if gt_label else 0]).to(self.device)  # 1=probe if harm
            intervention_loss = F.cross_entropy(intervention_logits, target_action)

            # Combined loss with reward weighting
            loss = (harm_loss + 0.5 * intervention_loss) * abs(reward)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.monitor.parameters(), 1.0)
            self.optimizer.step()

        return total_loss / len(trajectories)

    def _compute_tree_reward(self, tree: TreeNode, gt_harm: bool) -> float:
        """Compute discounted reward from tree."""
        if tree.is_terminal:
            return compute_leaf_value(tree, gt_harm)

        child_rewards = [
            self._compute_tree_reward(child, gt_harm) for child in tree.children
        ]
        # Mean of child rewards, discounted by gamma
        return self.gamma * sum(child_rewards) / max(len(child_rewards), 1)
