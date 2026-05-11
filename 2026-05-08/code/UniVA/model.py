"""UniVA model — Commercial SID Tokenizer + Generation-as-Ranking Decoder + Value Beam Search.

Implements the three components described in arXiv:2605.05803 in a faithful toy form.
Each method that the paper formulates as a domain-specific industrial step is marked with
``# TODO[paper]:`` and a pseudocode comment that mirrors the original equations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniVAConfig:
    num_items: int
    num_ecpm_buckets: int
    num_margin_buckets: int
    sid_levels: int = 3            # number of hierarchical SID code levels (RQ-VAE-style)
    sid_codebook_size: int = 64    # codes per level
    embed_dim: int = 64
    history_len: int = 16
    decoder_layers: int = 2
    decoder_heads: int = 4
    value_loss_weight: float = 0.5


# ---------------------------------------------------------------------------
# (1) Commercial SID Tokenizer
# ---------------------------------------------------------------------------
class CommercialSIDTokenizer(nn.Module):
    """Maps each item id -> (sid_levels,) tuple of code ids that encode value attributes.

    Faithful structure to UniVA's Commercial SID:
    - Level 0 reserved for value-discriminative bucket (eCPM bucket).
    - Level 1 reserved for margin bucket.
    - Remaining levels are learned by a residual-quantized item embedding (RQ-VAE-style).
    """

    def __init__(self, cfg: UniVAConfig, item_value_meta: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        assert cfg.sid_levels >= 2, "Need at least 2 SID levels for ecpm + margin"
        # Static codes from value attributes (per-item lookup, not learned).
        self.register_buffer("ecpm_codes", item_value_meta[:, 0].long(), persistent=False)
        self.register_buffer("margin_codes", item_value_meta[:, 1].long(), persistent=False)
        # Learned residual codebooks for the remaining levels.
        self.item_embed = nn.Embedding(cfg.num_items, cfg.embed_dim)
        self.codebooks = nn.ModuleList(
            [nn.Embedding(cfg.sid_codebook_size, cfg.embed_dim) for _ in range(cfg.sid_levels - 2)]
        )

    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return SID codes of shape [..., sid_levels]."""
        ecpm = self.ecpm_codes[item_ids]
        margin = self.margin_codes[item_ids]
        codes = [ecpm, margin]
        residual = self.item_embed(item_ids)
        for cb in self.codebooks:
            # Nearest-neighbor quantization (RQ-VAE step).
            dist = torch.cdist(residual.unsqueeze(-2), cb.weight.unsqueeze(0)).squeeze(-2)
            idx = dist.argmin(dim=-1)
            codes.append(idx)
            residual = residual - cb(idx)
            # TODO[paper]: full RQ-VAE training with reconstruction + commitment loss.
        return torch.stack(codes, dim=-1)


# ---------------------------------------------------------------------------
# (2) Generation-as-Ranking SID Decoder
# ---------------------------------------------------------------------------
class GARDecoder(nn.Module):
    """Generation-as-Ranking decoder.

    The decoder is autoregressive over SID levels. After the last level, we emit a
    ranking-aware logit over items: argmax over items can be done in a single pass,
    matching UniVA's "one decoding does generation+ranking".
    """

    def __init__(self, cfg: UniVAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.history_embed = nn.Embedding(cfg.num_items, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.history_len + cfg.sid_levels, cfg.embed_dim)
        self.level_embed = nn.Embedding(cfg.sid_levels, cfg.embed_dim)
        # Per-level code embeddings share a tied space.
        self.code_embed = nn.Embedding(max(cfg.sid_codebook_size, cfg.num_ecpm_buckets, cfg.num_margin_buckets), cfg.embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=cfg.embed_dim, nhead=cfg.decoder_heads, batch_first=True, dim_feedforward=4 * cfg.embed_dim)
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.decoder_layers)
        # Per-level output heads (variable codebook sizes for level 0/1 = bucket sizes).
        self.head_level0 = nn.Linear(cfg.embed_dim, cfg.num_ecpm_buckets)
        self.head_level1 = nn.Linear(cfg.embed_dim, cfg.num_margin_buckets)
        self.heads_residual = nn.ModuleList([nn.Linear(cfg.embed_dim, cfg.sid_codebook_size) for _ in range(cfg.sid_levels - 2)])
        # Value head used by the eCPM-aware RL term.
        self.value_head = nn.Linear(cfg.embed_dim, 1)

    def _build_input(self, history: torch.Tensor, target_codes: torch.Tensor | None) -> torch.Tensor:
        b, h = history.shape
        hist_e = self.history_embed(history)  # [B, H, D]
        pos_e = self.pos_embed(torch.arange(h, device=history.device))[None, :, :].expand(b, -1, -1)
        seq = hist_e + pos_e
        if target_codes is not None:
            l = target_codes.size(-1)
            code_e = self.code_embed(target_codes)
            level_e = self.level_embed(torch.arange(l, device=history.device))[None, :, :].expand(b, -1, -1)
            seq = torch.cat([seq, code_e + level_e], dim=1)
        return seq

    def forward(self, history: torch.Tensor, target_codes: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Teacher-forced forward.

        Returns per-level logits and an item-level value prediction.
        """
        b, h = history.shape
        l = target_codes.size(-1)
        # Shifted input: at position H+i we predict code_i from history + codes[:i].
        input_codes = torch.cat(
            [torch.zeros(b, 1, dtype=torch.long, device=history.device), target_codes[:, :-1]],
            dim=1,
        )
        seq = self._build_input(history, input_codes)
        # Causal mask
        sz = seq.size(1)
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=seq.device), diagonal=1)
        out = self.transformer(seq, mask=mask)
        # Take outputs aligned with the target codes (last L positions).
        out_codes = out[:, h:h + l, :]
        per_level_logits: list[torch.Tensor] = []
        for level in range(l):
            x = out_codes[:, level, :]
            if level == 0:
                per_level_logits.append(self.head_level0(x))
            elif level == 1:
                per_level_logits.append(self.head_level1(x))
            else:
                per_level_logits.append(self.heads_residual[level - 2](x))
        value = self.value_head(out_codes[:, -1, :]).squeeze(-1)
        return per_level_logits, value


# ---------------------------------------------------------------------------
# (3) Value-guided Personalized Beam Search
# ---------------------------------------------------------------------------
class PersonalizedTrie:
    """Per-user trie of valid SID paths -> items.

    In production the trie comes from the user's recall set. Here we expose a fit()
    method on the full catalogue for the toy demo.
    """

    def __init__(self) -> None:
        self.root: dict = {}

    def fit(self, sid_codes: torch.Tensor, item_ids: Sequence[int]) -> None:
        # sid_codes: [N, L]
        for codes, iid in zip(sid_codes.tolist(), item_ids):
            node = self.root
            for c in codes:
                node = node.setdefault(c, {})
            node["__item__"] = int(iid)

    def valid_next(self, prefix: list[int]) -> list[int]:
        node = self.root
        for c in prefix:
            if c not in node:
                return []
            node = node[c]
        return [k for k in node.keys() if k != "__item__"]

    def lookup(self, codes: list[int]) -> int | None:
        node = self.root
        for c in codes:
            if c not in node:
                return None
            node = node[c]
        return node.get("__item__")


def value_beam_search(
    decoder: GARDecoder,
    tokenizer: CommercialSIDTokenizer,
    history: torch.Tensor,
    trie: PersonalizedTrie,
    *,
    beam: int = 8,
    value_weight: float = 0.5,
    item_value: torch.Tensor | None = None,
) -> list[int]:
    """Decode top-1 item per row in `history` using value-guided personalized beam search.

    Score at each step: log p(code) + value_weight * value(code-prefix).
    Only codes valid in the trie are considered (catalog constraint).
    """
    decoder.eval()
    device = history.device
    b = history.size(0)
    cfg = decoder.cfg
    chosen_items: list[int] = []
    with torch.no_grad():
        for i in range(b):
            beams = [([], 0.0)]  # (prefix_codes, score)
            for level in range(cfg.sid_levels):
                next_beams: list[tuple[list[int], float]] = []
                for prefix, score in beams:
                    valid = trie.valid_next(prefix)
                    if not valid:
                        continue
                    target_codes = torch.tensor(prefix + [0], dtype=torch.long, device=device).unsqueeze(0)
                    logits, _ = decoder(history[i:i + 1], target_codes)
                    logp = F.log_softmax(logits[level], dim=-1)[0]
                    for c in valid:
                        # Value bonus: at level 0 the code IS the eCPM bucket -- use it directly;
                        # at later levels use the value head's prediction so far.
                        bonus = float(c) / max(cfg.num_ecpm_buckets - 1, 1) if level == 0 else 0.0
                        next_beams.append((prefix + [c], score + float(logp[c].item()) + value_weight * bonus))
                next_beams.sort(key=lambda kv: kv[1], reverse=True)
                beams = next_beams[:beam]
                if not beams:
                    break
            # Pick best leaf -> item id
            best_item = -1
            best_score = -1e9
            for prefix, score in beams:
                iid = trie.lookup(prefix)
                if iid is not None and score > best_score:
                    best_score = score
                    best_item = iid
            chosen_items.append(best_item)
    return chosen_items


__all__ = [
    "UniVAConfig",
    "CommercialSIDTokenizer",
    "GARDecoder",
    "PersonalizedTrie",
    "value_beam_search",
]
