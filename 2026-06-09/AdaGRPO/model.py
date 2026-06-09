from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SampledCandidate:
    seq: torch.LongTensor  # [L]
    logprob: torch.Tensor  # scalar


class GRPolicy(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        token_vocab: int,
        semantic_len: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.token_vocab = token_vocab
        self.semantic_len = semantic_len

        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.token_emb = nn.Embedding(token_vocab + 2, emb_dim)  # + {BOS, EOS}
        self.bos_id = token_vocab
        self.eos_id = token_vocab + 1

        self.ctx_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, token_vocab)

    def encode_history(self, history_item_ids: torch.LongTensor) -> torch.Tensor:
        # history_item_ids: [B, H]
        h = self.item_emb(history_item_ids)  # [B, H, emb]
        h = h.mean(dim=1)  # [B, emb]
        ctx = self.ctx_proj(h)  # [B, hidden]
        return ctx

    def nll(self, history_item_ids: torch.LongTensor, target_tokens: torch.LongTensor) -> torch.Tensor:
        """Teacher forcing NLL, returns mean loss per batch."""
        device = history_item_ids.device
        batch_size, length = target_tokens.shape
        assert length == self.semantic_len

        ctx = self.encode_history(history_item_ids)  # [B, hidden]
        h0 = ctx.unsqueeze(0)  # [1, B, hidden]

        inp = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        inp = torch.cat([inp, target_tokens[:, :-1]], dim=1)  # [B, L]
        emb = self.token_emb(inp)  # [B, L, emb]

        out, _ = self.gru(emb, h0)  # [B, L, hidden]
        logits = self.out(out)  # [B, L, V]

        loss = F.cross_entropy(logits.reshape(-1, self.token_vocab), target_tokens.reshape(-1), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(self, history_item_ids: torch.LongTensor, *, temperature: float = 1.0) -> tuple[torch.LongTensor, torch.Tensor]:
        """Sample one candidate per batch. Returns (seq[B,L], logprob[B])."""
        device = history_item_ids.device
        batch_size = history_item_ids.shape[0]

        ctx = self.encode_history(history_item_ids)
        h = ctx.unsqueeze(0)  # [1,B,H]

        prev = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        seq = []
        logps = torch.zeros((batch_size,), device=device)

        for _ in range(self.semantic_len):
            emb = self.token_emb(prev)  # [B,1,E]
            out, h = self.gru(emb, h)
            logits = self.out(out[:, 0, :])  # [B,V]
            logits = logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)  # [B,1]
            seq.append(tok)
            logps += torch.log(torch.gather(probs, 1, tok).squeeze(1) + 1e-12)
            prev = tok

        seq_t = torch.cat(seq, dim=1)  # [B,L]
        return seq_t, logps

    @torch.no_grad()
    def sample_k(self, history_item_ids: torch.LongTensor, *, k: int, temperature: float = 1.0) -> tuple[torch.LongTensor, torch.Tensor]:
        """Sample K candidates per sample.

        Returns:
          seq: [B, K, L]
          logprob: [B, K]
        """
        seqs = []
        lps = []
        for _ in range(k):
            s, lp = self.sample(history_item_ids, temperature=temperature)
            seqs.append(s.unsqueeze(1))
            lps.append(lp.unsqueeze(1))
        return torch.cat(seqs, dim=1), torch.cat(lps, dim=1)

    @torch.no_grad()
    def beam_search(self, history_item_ids: torch.LongTensor, *, beam_size: int = 10) -> tuple[torch.LongTensor, torch.Tensor]:
        """Beam search for fixed-length semantic ids.

        Returns:
          seq: [B, beam, L]
          logprob: [B, beam]
        """
        device = history_item_ids.device
        batch_size = history_item_ids.shape[0]

        ctx = self.encode_history(history_item_ids)  # [B,H]

        beams = torch.full((batch_size, 1, 0), 0, dtype=torch.long, device=device)
        beam_lp = torch.zeros((batch_size, 1), device=device)

        h = ctx.unsqueeze(0)  # [1,B,H]

        for step in range(self.semantic_len):
            # Expand each beam independently by running one GRU step from its last token.
            # For simplicity, we re-run from BOS each time (small L). This is fine for toy.
            all_new_seq = []
            all_new_lp = []

            for b in range(beams.shape[1]):
                prefix = beams[:, b, :]  # [B, step]

                inp = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
                if step > 0:
                    inp = torch.cat([inp, prefix], dim=1)  # [B, step+1]

                emb = self.token_emb(inp)  # [B, step+1, E]
                out, _ = self.gru(emb, h)  # starting hidden is ctx; OK for toy
                logits = self.out(out[:, -1, :])  # [B,V]
                logp = torch.log_softmax(logits, dim=-1)  # [B,V]

                topk_lp, topk_tok = torch.topk(logp, k=beam_size, dim=-1)  # [B,beam]
                base_lp = beam_lp[:, b].unsqueeze(1)  # [B,1]
                new_lp = base_lp + topk_lp  # [B,beam]

                new_seq = torch.cat([prefix.unsqueeze(1).expand(-1, beam_size, -1), topk_tok.unsqueeze(-1)], dim=-1)
                all_new_seq.append(new_seq)  # [B, beam, step+1]
                all_new_lp.append(new_lp)  # [B, beam]

            cand_seq = torch.cat(all_new_seq, dim=1)  # [B, beam*beam, step+1]
            cand_lp = torch.cat(all_new_lp, dim=1)  # [B, beam*beam]

            top_lp, top_idx = torch.topk(cand_lp, k=min(beam_size, cand_lp.shape[1]), dim=1)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, cand_seq.shape[-1])
            beams = torch.gather(cand_seq, 1, gather_idx)
            beam_lp = top_lp

        return beams, beam_lp


    def sequence_logprob(self, history_item_ids: torch.LongTensor, seq: torch.LongTensor) -> torch.Tensor:
        """Compute log P(seq | history) under teacher forcing.

        seq: [B, L]
        returns: [B]
        """
        device = history_item_ids.device
        batch_size, length = seq.shape
        assert length == self.semantic_len

        ctx = self.encode_history(history_item_ids)
        h0 = ctx.unsqueeze(0)

        inp = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        inp = torch.cat([inp, seq[:, :-1]], dim=1)
        emb = self.token_emb(inp)

        out, _ = self.gru(emb, h0)
        logits = self.out(out)  # [B,L,V]
        logp = torch.log_softmax(logits, dim=-1)
        tok_logp = torch.gather(logp, 2, seq.unsqueeze(-1)).squeeze(-1)  # [B,L]
        return tok_logp.sum(dim=1)

    def sequence_logprob_k(self, history_item_ids: torch.LongTensor, seq_k: torch.LongTensor) -> torch.Tensor:
        """seq_k: [B, K, L] -> logprob [B, K]."""
        b, k, l = seq_k.shape
        seq_flat = seq_k.reshape(b * k, l)
        hist_flat = history_item_ids.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        lp = self.sequence_logprob(hist_flat, seq_flat)
        return lp.reshape(b, k)
