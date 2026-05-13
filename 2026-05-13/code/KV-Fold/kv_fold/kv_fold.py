from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class KVFoldConfig:
    chunk_size: int = 256


def _make_attention_mask(
    input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if attention_mask is not None:
        return attention_mask
    return torch.ones_like(input_ids, dtype=torch.long)


def kv_fold_prefill(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cfg: KVFoldConfig = KVFoldConfig(),
) -> Tuple[PastKeyValues, torch.Tensor]:
    """Prefill a long prompt in chunks while carrying KV cache.

    Returns:
      - past_key_values after the full prompt is consumed
      - logits for the last consumed chunk
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, seq]")

    full_attention_mask = _make_attention_mask(input_ids, attention_mask)

    past_key_values: Optional[PastKeyValues] = None
    last_logits: Optional[torch.Tensor] = None

    seq_len = input_ids.shape[1]
    for start in range(0, seq_len, cfg.chunk_size):
        end = min(seq_len, start + cfg.chunk_size)
        chunk_ids = input_ids[:, start:end]
        chunk_attention = full_attention_mask[:, :end]

        outputs = model(
            input_ids=chunk_ids,
            attention_mask=chunk_attention,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        last_logits = outputs.logits

    if past_key_values is None or last_logits is None:
        raise RuntimeError("empty input")

    return past_key_values, last_logits


@torch.inference_mode()
def kv_fold_forward_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cfg: KVFoldConfig = KVFoldConfig(),
) -> torch.Tensor:
    """Forward a prompt with KV-Fold and return logits for all tokens."""

    full_attention_mask = _make_attention_mask(input_ids, attention_mask)

    past_key_values: Optional[PastKeyValues] = None
    seq_len = input_ids.shape[1]

    logits_chunks: List[torch.Tensor] = []

    for start in range(0, seq_len, cfg.chunk_size):
        end = min(seq_len, start + cfg.chunk_size)
        chunk_ids = input_ids[:, start:end]
        chunk_attention = full_attention_mask[:, :end]

        outputs = model(
            input_ids=chunk_ids,
            attention_mask=chunk_attention,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits_chunks.append(outputs.logits)

    return torch.cat(logits_chunks, dim=1)


@torch.inference_mode()
def kv_fold_greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    attention_mask: Optional[torch.Tensor] = None,
    cfg: KVFoldConfig = KVFoldConfig(),
) -> torch.Tensor:
    """Greedy-generate after KV-fold prefill.

    Notes:
      - This is a minimal implementation for the toy demo; it is not optimized.
    """

    full_attention_mask = _make_attention_mask(input_ids, attention_mask)
    past_key_values, _ = kv_fold_prefill(model, input_ids, full_attention_mask, cfg=cfg)

    generated = [input_ids]
    cur_input = input_ids[:, -1:]

    for step in range(max_new_tokens):
        cur_attention = torch.cat(
            [full_attention_mask, torch.ones((full_attention_mask.shape[0], step + 1), device=full_attention_mask.device, dtype=full_attention_mask.dtype)],
            dim=1,
        )

        outputs = model(
            input_ids=cur_input,
            attention_mask=cur_attention,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)
        cur_input = next_token

    return torch.cat(generated, dim=1)
