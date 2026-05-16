from __future__ import annotations

import dataclasses
import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_SID_L1_RE = re.compile(r"^SID_L1_(\d+)_(\d+)$")
_SID_L2_RE = re.compile(r"^SID_L2_(\d+)$")
_SID_L3_RE = re.compile(r"^SID_L3_(\d+)$")


@dataclasses.dataclass
class RqVaeConfig:
    n_categories: int
    latent_dim: int = 64
    input_dim: int = 64

    # 3-level residual quantization
    k1: int = 16
    k2: int = 32
    k3: int = 32

    decay: float = 0.99
    eps: float = 1e-5


class CategoryAwareResidualVQ(nn.Module):
    """A simplified RQ-VAE style residual quantizer with EMA codebook updates.

    This is a toy implementation focusing on the paper's key idea:
    - Level-1 codebook is **category-aware**.
    - Deeper levels use shared codebooks.
    """

    def __init__(self, cfg: RqVaeConfig):
        super().__init__()
        self.cfg = cfg

        # Codebooks
        self.codebook_l1 = nn.Parameter(torch.randn(cfg.n_categories, cfg.k1, cfg.latent_dim))
        self.codebook_l2 = nn.Parameter(torch.randn(cfg.k2, cfg.latent_dim))
        self.codebook_l3 = nn.Parameter(torch.randn(cfg.k3, cfg.latent_dim))

        # EMA buffers
        self.register_buffer("ema_count_l1", torch.zeros(cfg.n_categories, cfg.k1))
        self.register_buffer("ema_sum_l1", torch.zeros(cfg.n_categories, cfg.k1, cfg.latent_dim))

        self.register_buffer("ema_count_l2", torch.zeros(cfg.k2))
        self.register_buffer("ema_sum_l2", torch.zeros(cfg.k2, cfg.latent_dim))

        self.register_buffer("ema_count_l3", torch.zeros(cfg.k3))
        self.register_buffer("ema_sum_l3", torch.zeros(cfg.k3, cfg.latent_dim))

    @staticmethod
    def _nearest(x: torch.Tensor, codebook: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D], codebook: [K, D]
        # returns indices [B], vectors [B, D]
        dists = (
            (x.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(dim=-1)
        )  # [B, K]
        idx = dists.argmin(dim=-1)
        vec = codebook[idx]
        return idx, vec

    def _ema_update(self, idx: torch.Tensor, x: torch.Tensor, ema_count: torch.Tensor, ema_sum: torch.Tensor):
        # idx: [B], x: [B, D]
        # ema_count: [K], ema_sum: [K, D]
        one_hot = F.one_hot(idx, num_classes=ema_count.shape[0]).type_as(x)  # [B,K]
        batch_count = one_hot.sum(dim=0)  # [K]
        batch_sum = one_hot.t() @ x  # [K,D]

        ema_count.mul_(self.cfg.decay).add_(batch_count * (1 - self.cfg.decay))
        ema_sum.mul_(self.cfg.decay).add_(batch_sum * (1 - self.cfg.decay))

    @torch.no_grad()
    def _normalize_codebook(self, codebook: nn.Parameter, ema_count: torch.Tensor, ema_sum: torch.Tensor):
        n = ema_count.sum()
        # Laplace smoothing
        smoothed = (ema_count + self.cfg.eps) / (n + ema_count.numel() * self.cfg.eps) * n
        codebook.data.copy_(ema_sum / smoothed.unsqueeze(-1))

    def forward(self, z: torch.Tensor, category: torch.Tensor, update_ema: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize latent z.

        Args:
            z: [B, D]
            category: [B]
        Returns:
            z_q: [B, D] quantized
            info: dict with code indices
        """
        B, D = z.shape

        # Level 1: per-category codebooks
        idx1 = torch.empty((B,), device=z.device, dtype=torch.long)
        e1 = torch.empty_like(z)
        for c in category.unique().tolist():
            mask = category == c
            _idx, _e = self._nearest(z[mask], self.codebook_l1[c])
            idx1[mask] = _idx
            e1[mask] = _e

            if update_ema:
                self._ema_update(_idx, z[mask], self.ema_count_l1[c], self.ema_sum_l1[c])

        r1 = z - e1

        idx2, e2 = self._nearest(r1, self.codebook_l2)
        r2 = r1 - e2
        idx3, e3 = self._nearest(r2, self.codebook_l3)

        if update_ema:
            self._ema_update(idx2, r1, self.ema_count_l2, self.ema_sum_l2)
            self._ema_update(idx3, r2, self.ema_count_l3, self.ema_sum_l3)

            self._normalize_codebook(self.codebook_l2, self.ema_count_l2, self.ema_sum_l2)
            self._normalize_codebook(self.codebook_l3, self.ema_count_l3, self.ema_sum_l3)
            for c in range(self.cfg.n_categories):
                self._normalize_codebook(self.codebook_l1[c], self.ema_count_l1[c], self.ema_sum_l1[c])

        z_q = e1 + e2 + e3

        info = {
            "idx1": idx1,
            "idx2": idx2,
            "idx3": idx3,
        }
        return z_q, info


class RQVAE(nn.Module):
    def __init__(self, cfg: RqVaeConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(nn.Linear(cfg.input_dim, cfg.latent_dim), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.input_dim))
        self.quantizer = CategoryAwareResidualVQ(cfg)

    def forward(self, x: torch.Tensor, category: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encoder(x)
        z_q, info = self.quantizer(z, category=category, update_ema=self.training)

        # Straight-through estimator
        z_st = z + (z_q - z).detach()
        x_hat = self.decoder(z_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z, z_q.detach())
        loss = recon_loss + 0.25 * commit_loss

        info = dict(info)
        info.update({"recon_loss": recon_loss.detach(), "commit_loss": commit_loss.detach()})
        return loss, info

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, category: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        _, info = self.quantizer(z, category=category, update_ema=False)
        return info["idx1"], info["idx2"], info["idx3"]

    @torch.no_grad()
    def sid_tokens_from_indices(self, category: int, idx1: int, idx2: int, idx3: int) -> List[str]:
        return [f"SID_L1_{category}_{idx1}", f"SID_L2_{idx2}", f"SID_L3_{idx3}"]

    @torch.no_grad()
    def sid_embedding_from_tokens(self, sid_tokens: List[str]) -> torch.Tensor:
        """Map SID tokens back to a vector embedding.

        During RL sampling the decoder may emit invalid token types (e.g. repeating L1 tokens).
        For the toy reward we handle this gracefully:
        - pick the first token matching each level (L1/L2/L3)
        - if any level is missing, return a zero vector (low cosine reward)
        """
        l1 = None
        l2 = None
        l3 = None
        for t in sid_tokens:
            if l1 is None:
                m = _SID_L1_RE.match(t)
                if m:
                    l1 = m
                    continue
            if l2 is None:
                m = _SID_L2_RE.match(t)
                if m:
                    l2 = m
                    continue
            if l3 is None:
                m = _SID_L3_RE.match(t)
                if m:
                    l3 = m
                    continue

        if not (l1 and l2 and l3):
            return torch.zeros((self.cfg.latent_dim,), device=self.quantizer.codebook_l2.device)

        cat = int(l1.group(1))
        i1 = int(l1.group(2))
        i2 = int(l2.group(1))
        i3 = int(l3.group(1))
        return self.quantizer.codebook_l1[cat, i1] + self.quantizer.codebook_l2[i2] + self.quantizer.codebook_l3[i3]


class Seq2SeqSidModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(256, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return self.tok_emb(x) + self.pos_emb(pos)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Teacher forcing: predict tgt[:,1:] from tgt[:,:-1]
        src_key_padding_mask = src.eq(self.pad_id)
        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        enc = self.encoder(self._add_pos(src), src_key_padding_mask=src_key_padding_mask)

        tgt_key_padding_mask = tgt_inp.eq(self.pad_id)
        # causal mask
        T = tgt_inp.shape[1]
        causal = torch.triu(torch.ones(T, T, device=src.device, dtype=torch.bool), diagonal=1)

        dec = self.decoder(
            self._add_pos(tgt_inp),
            enc,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.lm_head(dec)  # [B,T,V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1),
            ignore_index=self.pad_id,
        )
        return loss

    @torch.no_grad()
    def encode_query_vector(self, src: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src.eq(self.pad_id)
        enc = self.encoder(self._add_pos(src), src_key_padding_mask=src_key_padding_mask)
        # mean pooling
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        pooled = (enc * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    @torch.no_grad()
    def sample(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample 1 sequence per batch element.

        Returns:
            seq: [B, L]
            logp: [B] sum log-prob
        """
        B = src.shape[0]
        src_key_padding_mask = src.eq(self.pad_id)
        memory = self.encoder(self._add_pos(src), src_key_padding_mask=src_key_padding_mask)

        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        sum_logp = torch.zeros((B,), device=src.device)

        for _ in range(max_len - 1):
            tgt_key_padding_mask = seq.eq(self.pad_id)
            T = seq.shape[1]
            causal = torch.triu(torch.ones(T, T, device=src.device, dtype=torch.bool), diagonal=1)

            dec = self.decoder(
                self._add_pos(seq),
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.lm_head(dec[:, -1, :]) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            sum_logp += torch.log(torch.gather(probs, 1, next_id.unsqueeze(1)).squeeze(1).clamp_min(1e-12))
            seq = torch.cat([seq, next_id.unsqueeze(1)], dim=1)

            if (next_id == eos_id).all():
                break

        return seq, sum_logp


@torch.no_grad()
def decode_sid_tokens(vocab, seq: torch.Tensor, eos_token: str = "<eos>") -> List[str]:
    # seq includes <bos> ... <eos>
    out = []
    for tid in seq.tolist():
        tok = vocab.id_to_token[tid]
        if tok == "<bos>":
            continue
        if tok == eos_token:
            break
        out.append(tok)
    return out


def eg_grpo_loss(
    model: Seq2SeqSidModel,
    rqvae: RQVAE,
    vocab,
    src: torch.Tensor,
    gt_tgt: torch.Tensor,
    bos_id: int,
    eos_id: int,
    group_size: int = 6,
    max_len: int = 5,
) -> torch.Tensor:
    """A minimal EG-GRPO objective.

    - Sample (group_size-1) sequences from the current policy.
    - Inject the ground-truth sequence as the last "expert" sample.
    - Compute **relative advantages** within the group.
    - Update policy on sampled sequences (excluding the injected expert).

    Reward design (toy):
      reward = 0.7 * exact_match + 0.3 * cosine(query_vec, sid_vec)
    """

    device = src.device
    B = src.shape[0]

    with torch.no_grad():
        query_vec = model.encode_query_vector(src)  # [B, d_model]
        query_vec = query_vec[:, : rqvae.cfg.latent_dim]

    # Collect sequences and logprobs
    sampled_seq: List[torch.Tensor] = []
    sampled_logp: List[torch.Tensor] = []
    for _ in range(group_size - 1):
        seq, logp = model.sample(src, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
        sampled_seq.append(seq)
        sampled_logp.append(logp)

    # Inject expert (ground truth)
    expert_seq = gt_tgt
    sampled_seq.append(expert_seq)
    sampled_logp.append(torch.zeros((B,), device=device))

    # Rewards
    rewards = []
    for seq in sampled_seq:
        with torch.no_grad():
            sid_tokens_batch = [decode_sid_tokens(vocab, s)[:3] for s in seq]
            sid_vec = torch.stack([rqvae.sid_embedding_from_tokens(toks) for toks in sid_tokens_batch], dim=0)
            cos = F.cosine_similarity(query_vec, sid_vec, dim=-1)

            # exact match on first 3 SID tokens
            gt_tokens = [decode_sid_tokens(vocab, s)[:3] for s in gt_tgt]
            exact = torch.tensor(
                [1.0 if sid_tokens_batch[i] == gt_tokens[i] else 0.0 for i in range(B)],
                device=device,
            )

            r = 0.7 * exact + 0.3 * cos
            rewards.append(r)

    rewards_t = torch.stack(rewards, dim=1)  # [B, G]
    advantages = rewards_t - rewards_t.mean(dim=1, keepdim=True)

    # Recompute log-probs with gradient for sampled sequences
    # (exclude injected expert at the end)
    loss = torch.tensor(0.0, device=device)
    for g in range(group_size - 1):
        seq = sampled_seq[g]

        # Compute token-level log-prob sum under teacher-forced decoding.
        # For simplicity, we ignore padding and assume short sequences.
        tgt_inp = seq[:, :-1]
        tgt_out = seq[:, 1:]

        src_key_padding_mask = src.eq(model.pad_id)
        enc = model.encoder(model._add_pos(src), src_key_padding_mask=src_key_padding_mask)

        tgt_key_padding_mask = tgt_inp.eq(model.pad_id)
        T = tgt_inp.shape[1]
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        dec = model.decoder(
            model._add_pos(tgt_inp),
            enc,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = model.lm_head(dec)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = torch.gather(log_probs, dim=-1, index=tgt_out.unsqueeze(-1)).squeeze(-1)
        seq_logp = token_logp.sum(dim=1)

        loss = loss - (advantages[:, g].detach() * seq_logp).mean()

    return loss / max(group_size - 1, 1)
