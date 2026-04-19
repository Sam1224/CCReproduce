from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, head_scale: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B,T,D]
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v  # [B,H,T,hd]
        if head_scale is not None:
            out = out * head_scale.view(1, -1, 1, 1)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))

    def forward(self, x: torch.Tensor, head_scale: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), head_scale=head_scale)
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_layers: int = 3, num_heads: int = 4, ff_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(256, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.num_heads = num_heads

    def forward(self, input_ids: torch.Tensor, head_scale: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = self.pos(torch.arange(seq_len, device=input_ids.device)).unsqueeze(0)
        x = self.embed(input_ids) + pos
        for blk in self.blocks:
            x = blk(x, head_scale=head_scale)
        x = self.ln_f(x)
        return self.lm_head(x)
