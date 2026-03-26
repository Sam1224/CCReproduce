import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper: OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework
Summary: Enhances industrial generative search with latent reasoning and self-distillation.
Core: Thought-augmented query understanding + reasoning-internalized self-distillation + preference alignment.

Note:
This file mirrors the lightweight reproduction style used in Sam1224/CCReproduce.
"""


class OneSearchV2(nn.Module):
    def __init__(self, d_model=256, vocab_size=4096, nhead=8):
        super().__init__()
        self.query_encoder = nn.TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True)
        self.cot_projection = nn.Linear(d_model, vocab_size)

        self.decoder = nn.TransformerDecoderLayer(d_model, nhead=nhead, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.reward_scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, query_emb, context_emb, teacher_mode=True):
        q_hidden = self.query_encoder(query_emb)
        latent_thoughts = self.cot_projection(q_hidden.mean(dim=1))

        if teacher_mode:
            ctx = context_emb + q_hidden.mean(dim=1).unsqueeze(1)
        else:
            ctx = context_emb

        decoded = self.decoder(tgt=ctx, memory=q_hidden)
        logits = self.lm_head(decoded)
        reward = self.reward_scorer(decoded.mean(dim=1))

        return logits, latent_thoughts, reward


def r_drop_loss(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return (
        F.kl_div(F.log_softmax(p_logits, dim=-1), q, reduction="batchmean")
        + F.kl_div(F.log_softmax(q_logits, dim=-1), p, reduction="batchmean")
    ) / 2


def onesearch_v2_loss(logits, target_ids, reward, target_reward, thought_logits, thought_target_ids, teacher_logits=None):
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    cot_loss = F.cross_entropy(thought_logits, thought_target_ids)
    reward_loss = F.mse_loss(reward, target_reward)

    dist = 0.0
    if teacher_logits is not None:
        dist = r_drop_loss(logits, teacher_logits)

    return lm_loss + cot_loss + reward_loss + dist


if __name__ == "__main__":
    torch.manual_seed(0)

    bsz, seq, d = 2, 16, 256
    q = torch.randn(bsz, seq, d)
    c = torch.randn(bsz, seq, d)

    model = OneSearchV2(d_model=d)
    logits, thoughts, reward = model(q, c, teacher_mode=True)

    print("logits:", tuple(logits.shape))
    print("thoughts:", tuple(thoughts.shape))
    print("reward:", tuple(reward.shape))
    print("OneSearch-V2 reproduction structure complete.")
