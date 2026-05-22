"""Multi-task loss for Valley3 omni e-commerce model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import NUM_CLASSES


class ECommerceLoss(nn.Module):
    """Unified loss routing to task-specific objectives.

    Classification/VQA/moderation/audio_cls → Cross-entropy.
    Captioning → token-level cross-entropy (language modeling).
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        if isinstance(task, (list, tuple)):
            task = task[0]

        if task == "captioning":
            # logits: (B, seq, vocab), labels: (B, target_len)
            B, seq, V = logits.shape
            tgt_len = labels.size(1)
            # Align: predict tokens starting from position 0
            pred = logits[:, :tgt_len].reshape(-1, V)
            tgt = labels.reshape(-1)
            return F.cross_entropy(pred, tgt, ignore_index=0)
        else:
            # labels: (B,)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            return F.cross_entropy(logits, labels.long())
