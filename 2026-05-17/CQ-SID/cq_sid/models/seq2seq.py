from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Seq2SeqOutput:
    logits: torch.Tensor
    loss: torch.Tensor


class TinyTransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: torch.Tensor,
        pad_id: int,
    ) -> Seq2SeqOutput:
        # teacher forcing; predict next token
        src_len = src_ids.size(1)
        tgt_len = tgt_ids.size(1)
        src_pos = torch.arange(src_len, device=src_ids.device)
        tgt_pos = torch.arange(tgt_len, device=tgt_ids.device)
        src = self.src_emb(src_ids) + self.pos_emb(src_pos)[None, :, :]
        tgt = self.tgt_emb(tgt_ids) + self.pos_emb(tgt_pos)[None, :, :]

        # transformer masks
        src_key_padding = ~src_mask
        tgt_key_padding = ~tgt_mask
        tgt_causal = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            tgt.device
        )

        out = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
            tgt_mask=tgt_causal,
        )
        logits = self.lm_head(out)  # [B, T, V]

        # next-token loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tgt_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_id,
        )
        return Seq2SeqOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_len: int = 16,
        num_samples: int = 4,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        device = src_ids.device
        bsz = src_ids.size(0)
        src_len = src_ids.size(1)
        src_pos = torch.arange(src_len, device=device)
        src = self.src_emb(src_ids) + self.pos_emb(src_pos)[None, :, :]
        src_key_padding = ~src_mask

        # sample multiple sequences per input
        all_out = []
        for _ in range(num_samples):
            ys = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
            for _step in range(max_len - 1):
                tgt_len = ys.size(1)
                tgt_pos = torch.arange(tgt_len, device=device)
                tgt = self.tgt_emb(ys) + self.pos_emb(tgt_pos)[None, :, :]
                tgt_causal = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
                    device
                )
                out = self.transformer(
                    src,
                    tgt,
                    src_key_padding_mask=src_key_padding,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=src_key_padding,
                    tgt_mask=tgt_causal,
                )
                logits = self.lm_head(out[:, -1, :]) / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                ys = torch.cat([ys, next_token], dim=1)
                if torch.all(next_token.squeeze(1) == eos_id):
                    break
            # pad to max_len
            if ys.size(1) < max_len:
                pad = torch.full(
                    (bsz, max_len - ys.size(1)), pad_id, dtype=torch.long, device=device
                )
                ys = torch.cat([ys, pad], dim=1)
            all_out.append(ys)
        return torch.stack(all_out, dim=1)  # [B, S, T]
