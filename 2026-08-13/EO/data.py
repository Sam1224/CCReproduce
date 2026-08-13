from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


ARM_NAMES = [
    "educate",
    "empathize",
    "social_proof",
    "advisor_cta",
]
OBS_DIM = 8
BELIEF_DIM = 3
NUM_ARMS = len(ARM_NAMES)
STEP_INPUT_DIM = OBS_DIM + NUM_ARMS
RESISTANCE_INDEX = 3
FRICTION_INDEX = 4


@dataclass
class SessionState:
    readiness: float
    info_need: float
    trust_need: float
    friction: float
    resistance: float
    returning: float
    urgency: float
    complexity: float
    initial_readiness: float


class ToyEODataset(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        actions: torch.Tensor,
        belief_targets: torch.Tensor,
    ) -> None:
        self.sequences = sequences.float()
        self.lengths = lengths.long()
        self.actions = actions.long()
        self.belief_targets = belief_targets.float()

    def __len__(self) -> int:
        return int(self.actions.size(0))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": self.sequences[index],
            "length": self.lengths[index],
            "action": self.actions[index],
            "belief_target": self.belief_targets[index],
        }


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand(1, generator=generator).item())


def _noise(scale: float, generator: torch.Generator) -> float:
    return float(torch.randn(1, generator=generator).item()) * scale


def arm_one_hot(index: int | None) -> torch.Tensor:
    vector = torch.zeros(NUM_ARMS, dtype=torch.float32)
    if index is not None:
        vector[index] = 1.0
    return vector


def sample_session_state(generator: torch.Generator) -> SessionState:
    readiness = _clip(0.10 + 0.85 * _rand(generator))
    info_need = _clip(0.20 + 0.70 * _rand(generator))
    trust_need = _clip(0.15 + 0.75 * _rand(generator))
    returning = 1.0 if _rand(generator) < 0.30 + 0.45 * readiness else 0.0
    urgency = _clip(0.15 + 0.55 * readiness + 0.20 * _rand(generator))
    complexity = _clip(0.15 + 0.50 * info_need + 0.30 * trust_need + 0.10 * _rand(generator))
    friction = _clip(0.08 + 0.45 * _rand(generator) + 0.20 * complexity - 0.15 * readiness)
    resistance = _clip(0.15 + 0.45 * (1.0 - readiness) + 0.18 * trust_need + 0.20 * friction + _noise(0.03, generator))
    return SessionState(
        readiness=readiness,
        info_need=info_need,
        trust_need=trust_need,
        friction=friction,
        resistance=resistance,
        returning=returning,
        urgency=urgency,
        complexity=complexity,
        initial_readiness=readiness,
    )


def observe_state(state: SessionState, step: int, max_turns: int, generator: torch.Generator) -> torch.Tensor:
    stage = float(step) / max(1, max_turns - 1)
    readiness_signal = _clip(
        0.68 * state.readiness + 0.12 * state.returning + 0.10 * state.urgency - 0.18 * state.resistance + _noise(0.03, generator)
    )
    info_signal = _clip(0.75 * state.info_need + 0.20 * state.complexity + 0.05 * (1.0 - stage) + _noise(0.03, generator))
    trust_signal = _clip(0.72 * state.trust_need + 0.15 * state.friction + 0.05 * (1.0 - state.returning) + _noise(0.03, generator))
    return torch.tensor(
        [
            readiness_signal,
            info_signal,
            trust_signal,
            state.resistance,
            state.friction,
            stage,
            state.returning,
            state.urgency,
        ],
        dtype=torch.float32,
    )


def oracle_arm_scores(state: SessionState, step: int, max_turns: int) -> torch.Tensor:
    stage = float(step) / max(1, max_turns - 1)
    scores = torch.tensor(
        [
            1.15 * state.info_need + 0.30 * state.friction + 0.18 * (1.0 - stage) - 0.20 * state.resistance,
            0.95 * state.trust_need + 0.80 * state.resistance + 0.35 * state.friction - 0.10 * stage,
            0.60 * state.trust_need + 0.72 * state.readiness + 0.22 * stage + 0.15 * (1.0 - abs(state.resistance - 0.40)),
            1.35 * state.readiness - 1.25 * state.resistance - 0.75 * state.friction + 0.45 * stage - 0.40 * state.info_need - 0.15 * state.trust_need,
        ],
        dtype=torch.float32,
    )
    return scores


def advance_session(
    state: SessionState,
    arm: int,
    step: int,
    max_turns: int,
    generator: torch.Generator,
) -> bool:
    stage = float(step) / max(1, max_turns - 1)
    jitter = _noise(0.015, generator)

    if arm == 0:
        info_before = state.info_need
        relief = _clip(0.14 + 0.12 * info_before - 0.04 * state.friction)
        state.info_need = _clip(info_before - relief)
        state.readiness = _clip(state.readiness + 0.05 + 0.08 * relief - 0.02 * state.friction)
        state.resistance = _clip(state.resistance - (0.06 + 0.12 * info_before + 0.03 * state.returning) + 0.03 * state.friction + jitter)
        state.friction = _clip(state.friction - 0.05 - 0.05 * relief + 0.01 * state.complexity + jitter)
        return False

    if arm == 1:
        trust_before = state.trust_need
        calming = _clip(0.15 + 0.14 * trust_before - 0.03 * state.friction)
        state.trust_need = _clip(trust_before - calming)
        state.readiness = _clip(state.readiness + 0.05 + 0.06 * calming)
        state.resistance = _clip(state.resistance - (0.07 + 0.12 * calming + 0.03 * trust_before) + 0.02 * state.friction + jitter)
        state.friction = _clip(state.friction - 0.04 - 0.04 * calming + jitter)
        return False

    if arm == 2:
        readiness_before = state.readiness
        proof_gain = _clip(0.08 + 0.10 * state.trust_need + 0.12 * readiness_before)
        state.trust_need = _clip(state.trust_need - (0.08 + 0.06 * readiness_before))
        state.readiness = _clip(readiness_before + 0.08 + 0.10 * proof_gain - 0.03 * state.friction)
        state.resistance = _clip(state.resistance - (0.05 + 0.04 * proof_gain + 0.06 * readiness_before) + 0.03 * state.friction + jitter)
        state.friction = _clip(state.friction - 0.03 + 0.01 * state.complexity + jitter)
        return False

    alignment = state.readiness - state.resistance - 0.55 * state.friction + 0.35 * stage - 0.25 * state.info_need
    contact_prob = float(torch.sigmoid(torch.tensor(4.5 * alignment + 0.40)).item())
    contacted = _rand(generator) < contact_prob
    if contacted:
        state.resistance = _clip(state.resistance - 0.18 + jitter)
        state.friction = _clip(state.friction - 0.06 + jitter)
        return True

    too_early_penalty = 0.10 if alignment < 0.08 else -0.04
    state.resistance = _clip(state.resistance + too_early_penalty + 0.05 * state.friction + jitter)
    state.friction = _clip(state.friction + max(0.0, too_early_penalty) * 0.60 + jitter)
    state.readiness = _clip(state.readiness + (0.05 if alignment >= 0.08 else -0.03))
    return False


def build_logged_dataset(
    num_sessions: int = 1400,
    max_turns: int = 6,
    seed: int = 13,
) -> ToyEODataset:
    generator = torch.Generator().manual_seed(seed)
    sequences: List[torch.Tensor] = []
    lengths: List[int] = []
    actions: List[int] = []
    belief_targets: List[torch.Tensor] = []

    for _ in range(num_sessions):
        state = sample_session_state(generator)
        prefix: List[torch.Tensor] = []
        prev_action = arm_one_hot(None)

        for step in range(max_turns):
            obs = observe_state(state, step, max_turns, generator)
            step_input = torch.cat([obs, prev_action], dim=0)
            prefix.append(step_input)

            padded = torch.zeros(max_turns, STEP_INPUT_DIM, dtype=torch.float32)
            padded[: len(prefix)] = torch.stack(prefix)
            sequences.append(padded)
            lengths.append(len(prefix))
            actions.append(int(oracle_arm_scores(state, step, max_turns).argmax().item()))
            belief_targets.append(torch.tensor([state.readiness, state.info_need, state.trust_need], dtype=torch.float32))

            contacted = advance_session(state, actions[-1], step, max_turns, generator)
            prev_action = arm_one_hot(actions[-1])
            if contacted or state.resistance > 0.96:
                break

    return ToyEODataset(
        sequences=torch.stack(sequences),
        lengths=torch.tensor(lengths, dtype=torch.long),
        actions=torch.tensor(actions, dtype=torch.long),
        belief_targets=torch.stack(belief_targets),
    )


def create_dataloaders(
    batch_size: int = 64,
    num_sessions: int = 1400,
    max_turns: int = 6,
    seed: int = 13,
) -> Tuple[DataLoader, DataLoader]:
    dataset = build_logged_dataset(num_sessions=num_sessions, max_turns=max_turns, seed=seed)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(seed + 97)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_policy(
    policy,
    num_sessions: int = 256,
    max_turns: int = 6,
    seed: int = 101,
    device: torch.device | None = None,
) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    advisor_contacts = 0
    genuine_contacts = 0
    resistance_drop = 0.0

    for _ in range(num_sessions):
        state = sample_session_state(generator)
        initial_resistance = state.resistance
        prefix: List[torch.Tensor] = []
        prev_action = arm_one_hot(None)
        contacted = False

        for step in range(max_turns):
            obs = observe_state(state, step, max_turns, generator)
            prefix.append(torch.cat([obs, prev_action], dim=0))
            padded = torch.zeros(max_turns, STEP_INPUT_DIM, dtype=torch.float32)
            padded[: len(prefix)] = torch.stack(prefix)
            action, _ = policy.act(padded if device is None else padded.to(device), len(prefix))
            contacted = advance_session(state, action, step, max_turns, generator)
            prev_action = arm_one_hot(action)
            if contacted or state.resistance > 0.96:
                break

        advisor_contacts += int(contacted)
        genuine_contacts += int(contacted and state.readiness > 0.72 and state.resistance < 0.35)
        resistance_drop += initial_resistance - state.resistance

    return {
        "advisor_contact_rate": advisor_contacts / max(1, num_sessions),
        "genuine_contact_rate": genuine_contacts / max(1, num_sessions),
        "avg_resistance_drop": resistance_drop / max(1, num_sessions),
    }
