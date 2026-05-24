from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchImageEncoder(nn.Module):
    def __init__(self, d_model: int = 256, patch: int = 8, image_size: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        grid = image_size // patch
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        x = self.proj(images)  # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, P, D]
        return x + self.pos


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        # token_ids: [B, T]
        key_padding_mask = token_ids.eq(pad_id)
        x = self.emb(token_ids)
        return self.enc(x, src_key_padding_mask=key_padding_mask)


class FineGrainedResidualEnhancement(nn.Module):
    """A lightweight FIRE-like block.

    In the paper, FIRE progressively preserves local details through deep propagation.
    Here we implement a simple gated residual MLP applied to the image patch tokens.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.mlp(x)


class MultiHeadModalityFusion(nn.Module):
    """A small fusion transformer over concatenated (image tokens + text tokens)."""

    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, img_tokens: torch.Tensor, txt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, p, d = img_tokens.shape
        _, t, _ = txt_tokens.shape

        img_type = self.type_emb(torch.zeros(bsz, p, dtype=torch.long, device=img_tokens.device))
        txt_type = self.type_emb(torch.ones(bsz, t, dtype=torch.long, device=img_tokens.device))

        x = torch.cat([img_tokens + img_type, txt_tokens + txt_type], dim=1)
        x = self.enc(x)
        return x[:, :p], x[:, p:]


class AttributeDeconstructor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor, pad_id: int) -> torch.Tensor:
        # Teacher forcing: predict next token.
        # tgt_ids: [B, L]
        bsz, seq_len = tgt_ids.shape
        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]

        tgt_key_padding = tgt_in.eq(pad_id)
        tgt = self.emb(tgt_in)

        # Causal mask
        causal = torch.triu(torch.ones(seq_len - 1, seq_len - 1, device=tgt.device), diagonal=1).bool()

        h = self.dec(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding,
        )
        logits = self.lm_head(h)  # [B, L-1, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_id,
        )
        return loss

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 20) -> torch.Tensor:
        device = memory.device
        bsz = memory.size(0)
        out = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt = self.emb(out)
            causal = torch.triu(torch.ones(out.size(1), out.size(1), device=device), diagonal=1).bool()
            h = self.dec(tgt, memory, tgt_mask=causal)
            logits = self.lm_head(h[:, -1])
            next_id = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return out


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.q.expand(tokens.size(0), -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)
        return pooled[:, 0]


@dataclass
class Moon3Outputs:
    embedding: torch.Tensor
    attr_loss: torch.Tensor


class Moon3ToyModel(nn.Module):
    def __init__(
        self,
        text_vocab_size: int,
        attr_vocab_size: int,
        text_pad_id: int,
        attr_pad_id: int,
        d_model: int = 256,
        embed_dim: int = 256,
        image_size: int = 64,
    ):
        super().__init__()
        self.text_pad_id = text_pad_id
        self.attr_pad_id = attr_pad_id

        self.img_enc = PatchImageEncoder(d_model=d_model, image_size=image_size)
        self.fire = FineGrainedResidualEnhancement(d_model)
        self.txt_enc = TextEncoder(vocab_size=text_vocab_size, d_model=d_model)
        self.fusion = MultiHeadModalityFusion(d_model=d_model)

        self.attr_head = AttributeDeconstructor(vocab_size=attr_vocab_size, d_model=d_model)
        self.pool = AttnPool(d_model)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, embed_dim))

    def encode(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        img = self.img_enc(images)
        img = self.fire(img)
        txt = self.txt_enc(text_ids, pad_id=self.text_pad_id)

        img_fused, txt_fused = self.fusion(img, txt)
        tokens = torch.cat([img_fused, txt_fused], dim=1)
        emb = self.proj(self.pool(tokens))
        return F.normalize(emb, dim=-1)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor, attr_ids: torch.Tensor) -> Moon3Outputs:
        img = self.img_enc(images)
        img = self.fire(img)
        txt = self.txt_enc(text_ids, pad_id=self.text_pad_id)
        img_fused, txt_fused = self.fusion(img, txt)
        memory = torch.cat([img_fused, txt_fused], dim=1)

        attr_loss = self.attr_head(attr_ids, memory=memory, pad_id=self.attr_pad_id)

        emb = self.proj(self.pool(memory))
        emb = F.normalize(emb, dim=-1)
        return Moon3Outputs(embedding=emb, attr_loss=attr_loss)

    @torch.no_grad()
    def generate_attributes(self, images: torch.Tensor, text_ids: torch.Tensor, bos_id: int, eos_id: int) -> torch.Tensor:
        img = self.fire(self.img_enc(images))
        txt = self.txt_enc(text_ids, pad_id=self.text_pad_id)
        img_fused, txt_fused = self.fusion(img, txt)
        memory = torch.cat([img_fused, txt_fused], dim=1)
        return self.attr_head.generate(memory, bos_id=bos_id, eos_id=eos_id)
