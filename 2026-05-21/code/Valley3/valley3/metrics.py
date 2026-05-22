"""Evaluation metrics for Valley3 tasks."""
import torch


def compute_task_metrics(logits: torch.Tensor, labels: torch.Tensor, task: str) -> dict:
    """Compute per-task accuracy.

    For classification/vqa/moderation/audio_cls: top-1 accuracy.
    For captioning: token-level accuracy.
    """
    if isinstance(task, (list, tuple)):
        task = task[0]

    if task == "captioning":
        B, seq, V = logits.shape
        tgt_len = labels.size(1)
        pred = logits[:, :tgt_len].argmax(-1)  # (B, tgt_len)
        correct = (pred == labels).float().mean()
        return {"token_acc": correct.item()}
    else:
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        pred = logits.argmax(-1)
        acc = (pred == labels.long()).float().mean()
        return {"accuracy": acc.item()}
