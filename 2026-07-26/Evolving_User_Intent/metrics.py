import torch
from torch.utils.data import DataLoader

from agent import action_slot_accuracy
from dataset import decode_actions


def intent_drift(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    intents = batch["intents"]
    initial = intents[:, 0:1, :]
    drift_from_initial = (intents != initial).float().mean(dim=-1)
    adjacent_drift = (intents[:, 1:, :] != intents[:, :-1, :]).float().mean(dim=-1)
    return {
        "avg_drift_from_initial": drift_from_initial.mean().item(),
        "final_drift_from_initial": drift_from_initial[:, -1].mean().item(),
        "avg_adjacent_drift": adjacent_drift.mean().item() if adjacent_drift.numel() else 0.0,
    }


def evaluate_agent(agent, loader: DataLoader, device: torch.device | str = "cpu") -> dict[str, float]:
    total_tokens = 0
    static_correct = 0
    evolving_correct = 0
    final_correct = 0
    drift_turn_correct = 0
    drift_turn_total = 0
    slot_score = 0.0
    drift_sums = {"avg_drift_from_initial": 0.0, "final_drift_from_initial": 0.0, "avg_adjacent_drift": 0.0}
    batches = 0

    for batch in loader:
        labels = batch["action_labels"].to(device)
        predictions = agent.predict(batch, device=device)
        correct = predictions == labels
        total_tokens += labels.numel()
        evolving_correct += correct.sum().item()
        static_correct += (predictions[:, 0] == labels[:, 0]).sum().item()
        final_correct += correct[:, -1].sum().item()
        slot_score += action_slot_accuracy(predictions, labels).sum().item()

        drift_mask = batch["drift_events"].to(device) > 0
        drift_turn_total += drift_mask.sum().item()
        if drift_mask.any():
            drift_turn_correct += correct[drift_mask].sum().item()

        drift = intent_drift(batch)
        for key, value in drift.items():
            drift_sums[key] += value
        batches += 1

    metrics = {
        "static_acc": static_correct / max(1, len(loader.dataset)),
        "evolving_acc": evolving_correct / max(1, total_tokens),
        "final_acc": final_correct / max(1, len(loader.dataset)),
        "drift_turn_acc": drift_turn_correct / max(1, drift_turn_total),
        "slot_acc": slot_score / max(1, total_tokens),
    }
    for key, value in drift_sums.items():
        metrics[key] = value / max(1, batches)
    metrics["static_to_evolving_gap"] = metrics["static_acc"] - metrics["evolving_acc"]
    return metrics


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    fields = [
        f"model={name}",
        f"static_acc={metrics['static_acc']:.3f}",
        f"evolving_acc={metrics['evolving_acc']:.3f}",
        f"final_acc={metrics['final_acc']:.3f}",
        f"drift_turn_acc={metrics['drift_turn_acc']:.3f}",
        f"slot_acc={metrics['slot_acc']:.3f}",
        f"gap={metrics['static_to_evolving_gap']:.3f}",
        f"final_drift={metrics['final_drift_from_initial']:.3f}",
    ]
    return " | ".join(fields)


def confusion_by_drift(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred_slots = decode_actions(predictions)
    label_slots = decode_actions(labels)
    return {
        "exact": (predictions == labels).float().mean().item(),
        "slot": (pred_slots == label_slots).float().mean().item(),
    }
