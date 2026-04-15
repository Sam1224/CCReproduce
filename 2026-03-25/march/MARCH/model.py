from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


@dataclass
class MarchOutput:
    response_logits: torch.Tensor  # (B,Lr,V)
    response_hidden: torch.Tensor  # (B,Lr,d)
    claim_pos: torch.Tensor  # (B,K)
    claim_logprob: torch.Tensor  # (B,K)
    checker_logits: torch.Tensor  # (B,K)


class Proposer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, response_hidden: torch.Tensor, response_mask: torch.Tensor, num_claims: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # response_hidden: (B,L,d)
        logits = self.score(response_hidden).squeeze(-1)  # (B,L)
        logits = logits.masked_fill(~response_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

        # Sample claim positions without replacement.
        claim_pos = torch.multinomial(probs, num_samples=num_claims, replacement=False)  # (B,K)

        # log p(pos)
        claim_logprob = torch.gather(torch.log(probs.clamp_min(1e-8)), 1, claim_pos)
        return claim_pos, claim_logprob


class Checker(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, claim_vec: torch.Tensor, evidence_vec: torch.Tensor) -> torch.Tensor:
        # claim_vec: (B,K,d), evidence_vec: (B,d)
        e = evidence_vec.unsqueeze(1).expand_as(claim_vec)
        logit = self.net(torch.cat([claim_vec, e], dim=-1)).squeeze(-1)
        return logit


class OneModel(nn.Module):
    """MARCH: Multi-Agent Reinforced Self-Check (toy).

    Key structure:
    - Solver generates response.
    - Proposer selects atomic claims (positions) from the response.
    - Checker verifies each claim against evidence in an information-asymmetric setting.
    - RL-style reward can be used to update solver/proposer.
    """

    def __init__(
        self,
        vocab_size: int = 6000,
        d_model: int = 256,
        nhead: int = 8,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_claims: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_claims = num_claims

        self.embed = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.proposer = Proposer(d_model)
        self.checker = Checker(d_model)

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).type_as(h)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (h * m).sum(dim=1) / denom

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> MarchOutput:
        query_ids = batch["query_ids"]
        query_mask = batch["query_mask"]
        evidence_ids = batch["evidence_ids"]
        evidence_mask = batch["evidence_mask"]
        response_in = batch["response_in"]
        response_mask = batch["response_mask"]

        # Encode [query; evidence] as memory.
        mem_ids = torch.cat([query_ids, evidence_ids], dim=1)
        mem_mask = torch.cat([query_mask, evidence_mask], dim=1)

        mem = self.embed(mem_ids).transpose(0, 1)  # (Lm,B,d)
        memory = self.encoder(mem, src_key_padding_mask=~mem_mask)

        # Decode response
        tgt = self.embed(response_in).transpose(0, 1)
        tgt_mask = _causal_mask(response_in.shape[1], tgt.device)
        dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        response_hidden = dec.transpose(0, 1)  # (B,Lr,d)
        response_logits = self.lm_head(dec).transpose(0, 1)  # (B,Lr,V)

        # Propose claim positions
        claim_pos, claim_logprob = self.proposer(response_hidden, response_mask, self.num_claims)

        # Checker: take claim vectors from response hidden, evidence vector from memory.
        claim_vec = torch.gather(
            response_hidden,
            dim=1,
            index=claim_pos.unsqueeze(-1).expand(-1, -1, response_hidden.shape[-1]),
        )
        memory_b = memory.transpose(0, 1)
        evidence_vec = self._masked_mean(memory_b, mem_mask)
        checker_logits = self.checker(claim_vec, evidence_vec)

        return MarchOutput(
            response_logits=response_logits,
            response_hidden=response_hidden,
            claim_pos=claim_pos,
            claim_logprob=claim_logprob,
            checker_logits=checker_logits,
        )


def march_losses(
    out: MarchOutput,
    response_tgt: torch.Tensor,
    claim_labels: torch.Tensor,
    reward_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    lm = F.cross_entropy(out.response_logits.reshape(-1, out.response_logits.shape[-1]), response_tgt.reshape(-1), ignore_index=-100)
    checker = F.binary_cross_entropy_with_logits(out.checker_logits, claim_labels)

    # REINFORCE-style loss on proposer: maximize reward (here: -checker loss per claim).
    with torch.no_grad():
        reward = -F.binary_cross_entropy_with_logits(out.checker_logits, claim_labels, reduction="none")  # (B,K)
    reinforce = -(reward_scale * reward * out.claim_logprob).mean()

    total = lm + checker + reinforce
    return total, {"lm_loss": float(lm.detach()), "checker_loss": float(checker.detach()), "reinforce": float(reinforce.detach())}
