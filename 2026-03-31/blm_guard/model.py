from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def aks_bin_top(scores: torch.Tensor, m: int = 4) -> torch.Tensor:
    """BIN+TOP keyframe selection.

    scores: (T,)
    """

    T = int(scores.shape[0])
    m = min(int(m), T)
    bins = m
    idx = []
    for b in range(bins):
        lo = int(math.floor(b * T / bins))
        hi = int(math.floor((b + 1) * T / bins))
        hi = max(lo + 1, hi)
        seg = scores[lo:hi]
        j = int(torch.argmax(seg).item()) + lo
        idx.append(j)
    idx = list(dict.fromkeys(idx))

    if len(idx) < m:
        rest = torch.argsort(scores, descending=True).tolist()
        for j in rest:
            if j not in idx:
                idx.append(j)
            if len(idx) >= m:
                break

    return torch.tensor(idx[:m], dtype=torch.long, device=scores.device)


class TransformerEncoder(nn.Module):
    def __init__(self, d: int, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=4 * d,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=key_padding_mask)
        return self.ln(x)


class CoTDecoder(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(256, d)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=4 * d,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.lm = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(ids) + self.pos(pos)

        tgt_mask = torch.triu(torch.ones(L, L, device=ids.device), diagonal=1).bool()

        for blk in self.blocks:
            x = blk(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)

        x = self.ln(x)
        return self.lm(x)

    @torch.no_grad()
    def generate(
        self,
        *,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        bos_id: int,
        eos_id: int,
        max_len: int = 96,
        temperature: float = 0.9,
        top_p: float = 0.92,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = memory.device
        B = int(memory.shape[0])
        ids = torch.full((B, 1), bos_id, device=device, dtype=torch.long)
        logp_sum = torch.zeros(B, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(ids, memory, memory_key_padding_mask=memory_key_padding_mask)[:, -1, :]
            logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)

            # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf <= top_p
            mask[:, 0] = True
            filtered = sorted_probs * mask
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)

            next_in_sorted = torch.multinomial(filtered, 1).squeeze(1)
            next_id = sorted_idx[torch.arange(B, device=device), next_in_sorted]

            lp = torch.log(probs.gather(1, next_id.unsqueeze(1)).squeeze(1).clamp_min(1e-9))
            logp_sum = logp_sum + lp

            ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
            if (next_id == eos_id).all():
                break

        return ids, logp_sum


@dataclass
class ModerationOutput:
    scene_logits: torch.Tensor
    type_logits: torch.Tensor
    risky_logits: torch.Tensor
    memory: torch.Tensor
    memory_pad_mask: torch.Tensor


class BLMGuardModel(nn.Module):
    def __init__(
        self,
        *,
        vocab: int,
        d_frame: int = 64,
        d_patch: int = 32,
        d: int = 256,
        n_scene: int = 7,
        n_type: int = 5,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.text_emb = nn.Embedding(vocab, d)
        self.text_pos = nn.Embedding(256, d)
        self.frame_proj = nn.Linear(d_frame, d)
        self.patch_proj = nn.Linear(d_patch, d)

        self.encoder = TransformerEncoder(d, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

        self.scene = nn.Linear(d, n_scene)
        self.type_head = nn.Linear(d, n_type)
        self.risky = nn.Linear(d, 2)

        self.decoder = CoTDecoder(vocab, d, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

    def _encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        return self.text_emb(ids) + self.text_pos(pos)

    def forward(
        self,
        *,
        frames: torch.Tensor,
        patches: torch.Tensor,
        frame_mask: torch.Tensor,
        asr_ids: torch.Tensor,
        ocr_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> ModerationOutput:
        B, T, _ = frames.shape

        # Risk scoring per frame (cheap heuristic): norm of frame embedding.
        score = frames.norm(dim=-1)  # (B,T)

        # BIN+TOP selection
        sel_idx = []
        for b in range(B):
            sel = aks_bin_top(score[b], m=min(6, T))
            sel_idx.append(sel)
        sel_idx = torch.stack(sel_idx, dim=0)  # (B,m)

        # Selected frames
        m = int(sel_idx.shape[1])
        sel_frames = frames.gather(1, sel_idx.unsqueeze(-1).expand(B, m, frames.shape[-1]))
        sel_patches = patches.gather(1, sel_idx[:, :, None, None].expand(B, m, patches.shape[2], patches.shape[3]))

        # Patch saliency selection (soft gating)
        patch_feat = self.patch_proj(sel_patches)  # (B,m,P,d)
        sal = patch_feat.norm(dim=-1)  # (B,m,P)
        w = torch.softmax(sal, dim=-1).unsqueeze(-1)
        patch_pooled = (patch_feat * w).sum(dim=2)  # (B,m,d)

        frame_tok = self.frame_proj(sel_frames) + patch_pooled  # (B,m,d)

        # Text tokens
        asr_tok = self._encode_text(asr_ids)
        ocr_tok = self._encode_text(ocr_ids)
        prm_tok = self._encode_text(prompt_ids)

        # Concatenate modalities
        x = torch.cat([prm_tok, asr_tok, ocr_tok, frame_tok], dim=1)  # (B, Ltot, d)

        # Build key padding mask (True means pad)
        def pad_mask(ids: torch.Tensor) -> torch.Tensor:
            return ids == 0

        m_prm = pad_mask(prompt_ids)
        m_asr = pad_mask(asr_ids)
        m_ocr = pad_mask(ocr_ids)
        m_vid = torch.zeros((B, m), device=frames.device, dtype=torch.bool)

        key_padding_mask = torch.cat([m_prm, m_asr, m_ocr, m_vid], dim=1)

        h = self.encoder(x, key_padding_mask=key_padding_mask)
        cls = h[:, 0, :]

        return ModerationOutput(
            scene_logits=self.scene(cls),
            type_logits=self.type_head(cls),
            risky_logits=self.risky(cls),
            memory=h,
            memory_pad_mask=key_padding_mask,
        )


def logprob_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # logits: (B,L,V) predict next token at each position
    # ids: (B,L) including BOS..EOS; compute token logprob for ids[:,1:]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = ids[:, 1:]
    return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B,L-1)


def cross_entropy_lm(logits: torch.Tensor, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    # teacher forcing: predict ids[t+1]
    tgt = ids[:, 1:].contiguous()
    pred = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(pred.view(-1, pred.shape[-1]), tgt.view(-1), ignore_index=pad_id)
    return loss
