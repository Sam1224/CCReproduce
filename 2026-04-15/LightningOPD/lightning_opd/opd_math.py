from __future__ import annotations

import torch


def shift_logits_and_labels(
    logits: torch.Tensor,  # [B, T, V]
    input_ids: torch.LongTensor,  # [B, T]
) -> tuple[torch.Tensor, torch.LongTensor]:
    # next-token prediction
    return logits[:, :-1, :].contiguous(), input_ids[:, 1:].contiguous()


def logp_of_labels(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    # logits: [B, T, V], labels: [B, T]
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
