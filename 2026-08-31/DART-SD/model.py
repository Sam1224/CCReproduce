import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InteractionStep:
    tool: str
    observation: str
    action_tokens: Tuple[int, ...]
    informative_atom: Optional[str]


@dataclass
class InteractionState:
    atoms: Set[str]
    useless_count: int = 0

    def transition(self, step: InteractionStep) -> "InteractionState":
        next_atoms = set(self.atoms)
        if step.informative_atom and step.informative_atom not in next_atoms:
            next_atoms.add(step.informative_atom)
            return InteractionState(next_atoms, 0)
        return InteractionState(next_atoms, self.useless_count + 1)

    @property
    def is_main(self) -> bool:
        return self.useless_count == 0

    def key(self) -> Tuple[Tuple[str, ...], int]:
        return tuple(sorted(self.atoms)), self.useless_count


class InteractionStateTransitionGraph:
    def __init__(self) -> None:
        self.success_paths: List[List[InteractionState]] = []
        self.failed_paths: List[List[InteractionState]] = []
        self.success_continuations: Dict[Tuple[Tuple[str, ...], int], List[Tuple[int, ...]]] = {}

    @staticmethod
    def states_from_steps(steps: Sequence[InteractionStep]) -> List[InteractionState]:
        states = [InteractionState(set(), 0)]
        current_state = states[0]
        for step in steps:
            current_state = current_state.transition(step)
            states.append(current_state)
        return states

    def add_rollout(self, steps: Sequence[InteractionStep], success: bool) -> None:
        states = self.states_from_steps(steps)
        target_paths = self.success_paths if success else self.failed_paths
        target_paths.append(states)
        if success:
            for index, state in enumerate(states[:-1]):
                suffix_tokens: List[int] = []
                for recovery_step in steps[index:]:
                    suffix_tokens.extend(recovery_step.action_tokens)
                self.success_continuations.setdefault(state.key(), []).append(tuple(suffix_tokens))

    def success_reachable_keys(self, depth_allowance: int = 2) -> Set[Tuple[Tuple[str, ...], int]]:
        if not self.success_paths:
            return set()
        shortest_length = min(len(path) - 1 for path in self.success_paths)
        budget = shortest_length + depth_allowance
        reachable: Set[Tuple[Tuple[str, ...], int]] = set()
        for path in self.success_paths:
            for reverse_depth, state in enumerate(reversed(path)):
                if reverse_depth <= budget:
                    reachable.add(state.key())
        return reachable

    def find_ctb(self, student_steps: Sequence[InteractionStep], depth_allowance: int = 2) -> Tuple[int, InteractionState]:
        reachable = self.success_reachable_keys(depth_allowance)
        states = self.states_from_steps(student_steps)
        last_valid_index = 0
        for state_index, state in enumerate(states):
            if state.key() in reachable or state_index == 0:
                last_valid_index = state_index
                continue
            return state_index, states[last_valid_index]
        return len(states) - 1, states[last_valid_index]

    def retrieve_recovery(self, anchor: InteractionState) -> Tuple[int, ...]:
        candidates = self.success_continuations.get(anchor.key(), [])
        if not candidates:
            return tuple()
        return min(candidates, key=len)


class DARTSelfDistillationModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        return self.output(encoded)


def localized_recovery_loss(logits: torch.Tensor, labels: torch.Tensor, recovery_mask: torch.Tensor) -> torch.Tensor:
    token_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), reduction="none")
    shifted_mask = recovery_mask[:, 1:].reshape(-1).float()
    denominator = shifted_mask.sum().clamp_min(1.0)
    return (token_loss * shifted_mask).sum() / denominator


def build_recovery_sequence(prefix_tokens: Sequence[int], recovery_tokens: Sequence[int], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    merged_tokens = list(prefix_tokens) + list(recovery_tokens)
    merged_tokens = merged_tokens[:max_length]
    mask = [0] * min(len(prefix_tokens), len(merged_tokens)) + [1] * max(0, len(merged_tokens) - len(prefix_tokens))
    if len(merged_tokens) < max_length:
        padding = max_length - len(merged_tokens)
        merged_tokens.extend([0] * padding)
        mask.extend([0] * padding)
    return torch.tensor(merged_tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def rollout_to_training_example(graph: InteractionStateTransitionGraph, student_steps: Sequence[InteractionStep], max_length: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    ctb_index, anchor = graph.find_ctb(student_steps)
    prefix_tokens: List[int] = []
    for step in student_steps[:ctb_index]:
        prefix_tokens.extend(step.action_tokens)
    recovery_tokens = graph.retrieve_recovery(anchor)
    return build_recovery_sequence(prefix_tokens, recovery_tokens, max_length)
