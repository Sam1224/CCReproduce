import torch
from torch import nn

from dataset import decode_actions, encode_intents


class StaticInitialAgent:
    """Strong static baseline: obeys the initial complete intent and never updates it."""

    name = "static_initial"

    def __init__(self, num_values: int = 4) -> None:
        self.num_values = num_values

    def predict(self, batch: dict[str, torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
        labels = batch["action_labels"].to(device)
        initial = labels[:, 0:1]
        return initial.repeat(1, labels.shape[1])


class TurnOnlyAgent:
    """A myopic policy that reads the latest update but forgets earlier revisions."""

    name = "turn_only"

    def __init__(self, num_values: int = 4) -> None:
        self.num_values = num_values

    def predict(self, batch: dict[str, torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
        update_mask = batch["update_mask"].to(device)
        update_values = batch["update_values"].to(device)
        initial_intent = batch["intents"].to(device)[:, 0, :]
        predictions = []
        for turn in range(update_mask.shape[1]):
            current = initial_intent.clone()
            mask = update_mask[:, turn, :] > 0.5
            current[mask] = update_values[:, turn, :][mask]
            predictions.append(encode_intents(current, self.num_values))
        return torch.stack(predictions, dim=1)


class RuleMemoryTracker:
    """Oracle-style lightweight memory tracker using explicit slot updates."""

    name = "rule_memory_tracker"

    def __init__(self, num_values: int = 4) -> None:
        self.num_values = num_values

    def predict(self, batch: dict[str, torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
        update_mask = batch["update_mask"].to(device)
        update_values = batch["update_values"].to(device)
        memory = update_values[:, 0, :].clone()
        predictions = []
        for turn in range(update_mask.shape[1]):
            mask = update_mask[:, turn, :] > 0.5
            memory[mask] = update_values[:, turn, :][mask]
            predictions.append(encode_intents(memory, self.num_values))
        return torch.stack(predictions, dim=1)


class NeuralMemoryAgent(nn.Module):
    """Tiny GRU policy that must learn to maintain evolving user intent."""

    name = "neural_memory"

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_actions: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy = nn.Linear(hidden_dim, num_actions)

    def forward(self, utterances: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(utterances)
        hidden, _ = self.gru(encoded)
        return self.policy(hidden)

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
        self.eval()
        logits = self.forward(batch["utterances"].to(device))
        return logits.argmax(dim=-1)


def action_slot_accuracy(predictions: torch.Tensor, labels: torch.Tensor, num_slots: int = 3, num_values: int = 4) -> torch.Tensor:
    pred_slots = decode_actions(predictions, num_slots=num_slots, num_values=num_values)
    label_slots = decode_actions(labels, num_slots=num_slots, num_values=num_values)
    return (pred_slots == label_slots).float().mean(dim=-1)
