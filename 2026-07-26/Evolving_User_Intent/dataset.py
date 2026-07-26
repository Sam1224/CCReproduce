import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


SLOT_NAMES = ("topic", "tone", "budget")
VALUE_NAMES = (
    ("travel", "fitness", "finance", "cooking"),
    ("brief", "friendly", "formal", "technical"),
    ("cheap", "balanced", "premium", "urgent"),
)


@dataclass
class DialogueSample:
    utterances: torch.Tensor
    update_mask: torch.Tensor
    update_values: torch.Tensor
    intents: torch.Tensor
    action_labels: torch.Tensor
    drift_events: torch.Tensor


def encode_intents(intents: torch.Tensor, num_values: int = 4) -> torch.Tensor:
    """Convert [*, num_slots] slot values to a flat action id."""
    action = torch.zeros(intents.shape[:-1], dtype=torch.long, device=intents.device)
    for slot in range(intents.shape[-1]):
        action = action * num_values + intents[..., slot].long()
    return action


def decode_actions(actions: torch.Tensor, num_slots: int = 3, num_values: int = 4) -> torch.Tensor:
    """Inverse of encode_intents."""
    decoded = []
    value = actions.long()
    for _ in range(num_slots):
        decoded.append(value % num_values)
        value = value // num_values
    return torch.stack(list(reversed(decoded)), dim=-1)


class EvolvingIntentDataset(Dataset):
    """
    Synthetic multi-turn user-intent benchmark.

    Turn 0 states a complete intent over three slots. Later turns may revise one
    slot, creating the paper's evolving-intent setting where the correct answer
    is the *current* intent, not the initial instruction.
    """

    def __init__(
        self,
        samples: int = 512,
        turns: int = 6,
        num_slots: int = 3,
        num_values: int = 4,
        drift_prob: float = 0.7,
        seed: int = 17,
    ) -> None:
        self.samples = samples
        self.turns = turns
        self.num_slots = num_slots
        self.num_values = num_values
        self.drift_prob = drift_prob
        self.feature_dim = num_slots * num_values + num_slots + 2
        self.num_actions = num_values**num_slots
        self.data = self._build(seed)

    def _feature_from_update(
        self,
        update_mask: torch.Tensor,
        update_values: torch.Tensor,
        turn_index: int,
        drift_event: int,
    ) -> torch.Tensor:
        slot_value = torch.zeros(self.num_slots, self.num_values)
        for slot in range(self.num_slots):
            if update_mask[slot] > 0.5:
                slot_value[slot, update_values[slot].item()] = 1.0
        scalar = torch.tensor(
            [turn_index / max(1, self.turns - 1), float(drift_event)],
            dtype=torch.float,
        )
        return torch.cat([slot_value.flatten(), update_mask.float(), scalar], dim=0)

    def _build(self, seed: int) -> list[DialogueSample]:
        rng = random.Random(seed)
        dataset = []
        for _ in range(self.samples):
            current = torch.tensor([rng.randrange(self.num_values) for _ in range(self.num_slots)], dtype=torch.long)
            utterances = []
            masks = []
            values = []
            intents = []
            drift_events = []

            for turn in range(self.turns):
                if turn == 0:
                    mask = torch.ones(self.num_slots)
                    update_value = current.clone()
                    drift_event = 0
                else:
                    should_drift = rng.random() < self.drift_prob
                    mask = torch.zeros(self.num_slots)
                    update_value = torch.zeros(self.num_slots, dtype=torch.long)
                    drift_event = int(should_drift)
                    if should_drift:
                        slot = rng.randrange(self.num_slots)
                        candidates = [value for value in range(self.num_values) if value != current[slot].item()]
                        current[slot] = rng.choice(candidates)
                        mask[slot] = 1.0
                        update_value[slot] = current[slot]

                utterances.append(self._feature_from_update(mask, update_value, turn, drift_event))
                masks.append(mask)
                values.append(update_value)
                intents.append(current.clone())
                drift_events.append(drift_event)

            intents_tensor = torch.stack(intents)
            dataset.append(
                DialogueSample(
                    utterances=torch.stack(utterances),
                    update_mask=torch.stack(masks),
                    update_values=torch.stack(values),
                    intents=intents_tensor,
                    action_labels=encode_intents(intents_tensor, self.num_values),
                    drift_events=torch.tensor(drift_events, dtype=torch.long),
                )
            )
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.data[index]
        return {
            "utterances": sample.utterances.clone(),
            "update_mask": sample.update_mask.clone(),
            "update_values": sample.update_values.clone(),
            "intents": sample.intents.clone(),
            "action_labels": sample.action_labels.clone(),
            "drift_events": sample.drift_events.clone(),
        }

    def render_dialogue(self, index: int = 0) -> list[str]:
        sample = self.data[index]
        lines = []
        for turn in range(self.turns):
            updates = []
            for slot in range(self.num_slots):
                if sample.update_mask[turn, slot] > 0.5:
                    name = SLOT_NAMES[slot]
                    value = VALUE_NAMES[slot][sample.update_values[turn, slot].item()]
                    updates.append(f"{name}={value}")
            if not updates:
                updates.append("no explicit change")
            current = ", ".join(
                f"{SLOT_NAMES[slot]}={VALUE_NAMES[slot][sample.intents[turn, slot].item()]}"
                for slot in range(self.num_slots)
            )
            lines.append(f"turn {turn}: update({'; '.join(updates)}) -> current intent({current})")
        return lines


if __name__ == "__main__":
    dataset = EvolvingIntentDataset(samples=2, turns=5, drift_prob=0.8)
    print(dataset[0]["utterances"].shape)
    print(dataset[0]["action_labels"])
    for line in dataset.render_dialogue(0):
        print(line)
