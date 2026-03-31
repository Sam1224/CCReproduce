from __future__ import annotations

import torch


class MemorySparseAttention(torch.nn.Module):
    def __init__(self, *, d: int = 64, heads: int = 4, topk: int = 8, num_classes: int = 12):
        super().__init__()
        self.d = d
        self.topk = topk

        self.score = torch.nn.Sequential(
            torch.nn.Linear(2 * d, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, 1),
        )

        self.attn = torch.nn.MultiheadAttention(embed_dim=d, num_heads=heads, batch_first=True)
        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(d),
            torch.nn.Linear(d, 2 * d),
            torch.nn.GELU(),
            torch.nn.Linear(2 * d, d),
        )
        self.cls = torch.nn.Linear(d, num_classes)

    def forward(self, docs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """docs: (M, D), query: (D,) -> logits: (C,)"""
        M, D = docs.shape
        q = query.view(1, 1, D).expand(1, M, D)
        x = torch.cat([docs.view(1, M, D), q], dim=-1)

        # (1, M)
        scores = self.score(x).squeeze(-1)

        # soft distribution (differentiable) + hard top-k for sparse attention
        weights = torch.softmax(scores, dim=-1)
        k = min(self.topk, M)
        top_idx = torch.topk(scores, k=k, dim=-1).indices.squeeze(0)
        mem = docs[top_idx].unsqueeze(0)  # (1, K, D)

        q_token = query.view(1, 1, D)
        attn_out, _ = self.attn(q_token, mem, mem, need_weights=False)
        # mix in soft global summary as a cheap 'linear' signal
        global_sum = (weights.view(1, M, 1) * docs.view(1, M, D)).sum(dim=1, keepdim=True)

        h = attn_out + 0.2 * global_sum
        h = h + self.ffn(h)
        logits = self.cls(h.squeeze(0).squeeze(0))
        return logits
