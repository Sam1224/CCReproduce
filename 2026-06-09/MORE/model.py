from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqGRU(nn.Module):
    def __init__(self, *, vocab_size: int, emb_dim: int = 96, hidden_dim: int = 192) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.enc = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dec = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src: torch.LongTensor, src_len: torch.LongTensor) -> torch.Tensor:
        # src: [B,T]
        emb = self.emb(src)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h  # [1,B,H]

    def nll(self, src: torch.LongTensor, src_len: torch.LongTensor, tgt: torch.LongTensor) -> torch.Tensor:
        device = src.device
        bos = tgt[:, :1]
        inp = torch.cat([bos, tgt[:, 1:-1]], dim=1)  # shift right, keep eos as label

        h0 = self.encode(src, src_len)
        emb = self.emb(inp)
        out, _ = self.dec(emb, h0)
        logits = self.out(out)  # [B, L-1, V]

        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), tgt[:, 1:].reshape(-1), reduction="mean")
        return loss

    def sequence_logprob(self, src: torch.LongTensor, src_len: torch.LongTensor, seq: torch.LongTensor) -> torch.Tensor:
        """Compute log P(seq | src) for a sequence that includes BOS/EOS.

        seq: [B, L] with BOS at 0 and EOS somewhere at end.
        returns: [B]
        """
        bos = seq[:, :1]
        inp = torch.cat([bos, seq[:, 1:-1]], dim=1)

        h0 = self.encode(src, src_len)
        emb = self.emb(inp)
        out, _ = self.dec(emb, h0)
        logits = self.out(out)
        logp = torch.log_softmax(logits, dim=-1)
        tok_logp = torch.gather(logp, 2, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
        return tok_logp.sum(dim=1)

    @torch.no_grad()
    def generate(self, src: torch.LongTensor, src_len: torch.LongTensor, *, bos_id: int, eos_id: int, max_len: int = 64, temperature: float = 1.0) -> tuple[torch.LongTensor, torch.Tensor]:
        device = src.device
        bsz = src.shape[0]

        h = self.encode(src, src_len)
        prev = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)

        seq = [prev]
        lp = torch.zeros((bsz,), device=device)
        finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.emb(prev)
            out, h = self.dec(emb, h)
            logits = self.out(out[:, 0, :]) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            tok_prob = torch.gather(probs, 1, tok).squeeze(1)
            lp += torch.log(tok_prob + 1e-12) * (~finished)

            seq.append(tok)
            finished = finished | (tok.squeeze(1) == eos_id)
            prev = tok
            if bool(finished.all()):
                break

        seq_t = torch.cat(seq, dim=1)
        return seq_t, lp

    @torch.no_grad()
    def greedy(self, src: torch.LongTensor, src_len: torch.LongTensor, *, bos_id: int, eos_id: int, max_len: int = 64) -> torch.LongTensor:
        device = src.device
        bsz = src.shape[0]

        h = self.encode(src, src_len)
        prev = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
        seq = [prev]
        finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.emb(prev)
            out, h = self.dec(emb, h)
            tok = torch.argmax(self.out(out[:, 0, :]), dim=-1, keepdim=True)
            seq.append(tok)
            finished = finished | (tok.squeeze(1) == eos_id)
            prev = tok
            if bool(finished.all()):
                break

        return torch.cat(seq, dim=1)
