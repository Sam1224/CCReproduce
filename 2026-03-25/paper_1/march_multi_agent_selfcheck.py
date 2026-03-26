import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper: MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination
Summary: Information-asymmetric multi-agent pipeline to reduce hallucination in RAG.
Core: Solver (generate) -> Proposer (atomize claims) -> Checker (blind verification), optimized via RL.

Note:
This is a minimal PyTorch reproduction of the *core structure* (roles + information asymmetry + reward shape),
not a full LLM/RAG system.
"""


class Solver(nn.Module):
    """Toy generator: produces token logits given query+evidence embeddings."""

    def __init__(self, d_model=256, vocab_size=4096, nhead=8):
        super().__init__()
        self.decoder = nn.TransformerDecoderLayer(d_model, nhead=nhead, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, query_emb, evidence_emb):
        hidden = self.decoder(tgt=query_emb, memory=evidence_emb)
        return self.lm_head(hidden)


class Proposer(nn.Module):
    """Toy atomizer: turns a response embedding into claim logits."""

    def __init__(self, d_model=256, claim_vocab=1024):
        super().__init__()
        self.claim_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, claim_vocab),
        )

    def forward(self, response_hidden):
        pooled = response_hidden.mean(dim=1)
        return self.claim_head(pooled)


class Checker(nn.Module):
    """Blind verifier: only sees evidence + atomic claim (no Solver output)."""

    def __init__(self, d_model=256, claim_vocab=1024, nhead=8):
        super().__init__()
        self.claim_emb = nn.Embedding(claim_vocab, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True),
            num_layers=2,
        )
        self.verdict = nn.Linear(d_model, 2)  # 0: unsupported, 1: supported

    def forward(self, evidence_emb, claim_id):
        claim_token = self.claim_emb(claim_id).unsqueeze(1)
        x = torch.cat([claim_token, evidence_emb], dim=1)
        h = self.encoder(x)
        return self.verdict(h[:, 0])


class MARCH(nn.Module):
    def __init__(self, d_model=256, vocab_size=4096, claim_vocab=1024):
        super().__init__()
        self.solver = Solver(d_model=d_model, vocab_size=vocab_size)
        self.proposer = Proposer(d_model=d_model, claim_vocab=claim_vocab)
        self.checker = Checker(d_model=d_model, claim_vocab=claim_vocab)

    def forward(self, query_emb, evidence_emb, claim_id=None):
        solver_logits = self.solver(query_emb, evidence_emb)
        response_hidden = torch.tanh(solver_logits[..., : query_emb.size(-1)])

        claim_logits = self.proposer(response_hidden)
        if claim_id is None:
            claim_id = claim_logits.argmax(dim=-1)

        checker_logits = self.checker(evidence_emb=evidence_emb, claim_id=claim_id)
        return solver_logits, claim_logits, checker_logits, claim_id


def march_reward(checker_logits, target_supported=1):
    """A strict, verifiable reward: penalize if a claim is judged unsupported."""

    target = torch.full(
        (checker_logits.size(0),),
        fill_value=int(target_supported),
        dtype=torch.long,
        device=checker_logits.device,
    )
    ce = F.cross_entropy(checker_logits, target)
    return -ce


if __name__ == "__main__":
    torch.manual_seed(0)

    bsz, q_len, ev_len, d = 2, 12, 24, 256
    query = torch.randn(bsz, q_len, d)
    evidence = torch.randn(bsz, ev_len, d)

    model = MARCH(d_model=d)
    solver_logits, claim_logits, checker_logits, claim_id = model(query, evidence)

    reward = march_reward(checker_logits)

    print("solver_logits:", tuple(solver_logits.shape))
    print("claim_id:", claim_id.tolist())
    print("reward:", reward.detach().cpu().tolist())
    print("MARCH reproduction structure complete.")
