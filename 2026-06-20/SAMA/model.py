"""SAMA model: Collaborative Multi-Experts MLLM (CME-MLLM).

Faithful (toy-scale, runnable) reproduction of SAMA's text-generation pipeline:
  Guo et al., "SAMA: Semantic Anchor-aligned Augmentation for Unified Low-Resource
  Multimodal Information Extraction" (UESTC).

Components (paper Sec III.B-D):
  * LoRA adapters (Eq. 1): h = W0 x + B A x  -> structural basis for the experts.
  * InstructBLIP-style Q-Former stub: learnable queries cross-attend image patches.
  * Anchor fusion s_anchor = f_InstructBLIP(T, I, A_n)  (Eq. 2).
  * Collaborative experts: Universal U(x) + Task-Specific D_n(x), combined by an
    anchor-motivated gate  g = Softmax(Wg [s_anchor, xt]);  h = gu U(x) + gd D_n(x)
    (Eqs. 3-4), then vocab projection P(w) = Softmax(W_head h) (Eq. 5).
  * Knowledge harmonization (Eq. 6): MI(U; D_n) maximised via InfoNCE
    (teacher=Universal, student=Task-Specific).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchors import anchor_ids, entity_anchor_words
from data import (GOLD2ID, IMG_C, IMG_H, IMG_W, STOI, TASKS, VOCAB, Sample,
                  encode)

D = 32                # hidden size
RANK = 4              # LoRA rank r << d
N_QUERY = 4           # Q-Former query tokens
PAD = STOI["<pad>"]
V = len(VOCAB)


# ---- LoRA adapter (Eq. 1) ----------------------------------------------------
class LoRA(nn.Module):
    """Low-rank delta: returns (B A x). Used as Universal / Task-Specific expert."""

    def __init__(self, d=D, r=RANK, alpha=2.0):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)            # delta starts at 0 (LoRA convention)
        self.scale = alpha / r

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


# ---- image encoder + Q-Former stub ------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(IMG_C, D, kernel_size=4, stride=4)   # [3,8,8]->[D,2,2]

    def forward(self, img):                  # img [B,3,8,8]
        h = self.conv(img)                   # [B,D,2,2]
        return h.flatten(2).transpose(1, 2)  # [B,4,D] patch tokens


class QFormer(nn.Module):
    """InstructBLIP-style: learnable queries cross-attend to image patch tokens."""

    def __init__(self, n_query=N_QUERY):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, D) * 0.02)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)

    def forward(self, patches):              # [B,4,D]
        B = patches.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, patches, patches)
        return out                            # [B,N_QUERY,D] visual tokens


# ---- text backbone (causal LM) ----------------------------------------------
class TextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 64, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=64,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, ids, prefix):          # ids [B,L]; prefix [B,P,D] visual+anchor
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        h = torch.cat([prefix, x], dim=1)    # prepend multimodal prefix
        P = prefix.size(1)
        Ltot = h.size(1)
        mask = torch.triu(torch.full((Ltot, Ltot), float("-inf")), diagonal=1)
        mask[:, :P] = 0.0                    # text can always attend to the prefix
        h = self.enc(h, mask=mask)
        return h[:, P:]                       # [B,L,D] hidden for text positions


# ---- CME-MLLM ----------------------------------------------------------------
class CME_MLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.qformer = QFormer()
        self.text = TextBackbone()
        self.anchor_proj = nn.Linear(2 * D, D)     # f_InstructBLIP fusion -> s_anchor
        # collaborative experts
        self.universal = LoRA()                    # U(x): shared across tasks
        self.task_experts = nn.ModuleDict({t: LoRA() for t in TASKS})  # D_n(x)
        # anchor-motivated gate: input [s_anchor(D), h_text(D), h_image(D)] -> 2
        self.gate = nn.Linear(3 * D, 2)
        self.head = nn.Linear(D, V)                # W_head (Eq. 5)

    def encode_image(self, img):
        return self.qformer(self.img_enc(img))     # [B,N_QUERY,D]

    def anchor_embedding(self, anchor_ids_b, vis_tokens):
        """s_anchor = f_InstructBLIP(T, I, A_n)  (Eq. 2)."""
        a_emb = self.text.emb(anchor_ids_b)                    # [B,La,D]
        a_pool = a_emb.mean(1)                                 # textual anchor
        v_pool = vis_tokens.mean(1)                            # visual context
        return torch.tanh(self.anchor_proj(torch.cat([a_pool, v_pool], -1)))  # [B,D]

    def forward(self, ids, img, anchor_ids_b, task: str):
        vis = self.encode_image(img)                          # [B,N_QUERY,D]
        s_anchor = self.anchor_embedding(anchor_ids_b, vis)   # [B,D]
        h = self.text(ids, vis)                               # [B,L,D] base hidden (= W0 x)
        B, L, _ = h.shape
        img_ctx = vis.mean(1, keepdim=True).expand(B, L, D)   # h_image^t
        gate_in = torch.cat([s_anchor.unsqueeze(1).expand(B, L, D), h, img_ctx], -1)
        g = F.softmax(self.gate(gate_in), dim=-1)             # [B,L,2] = [gu, gd]
        u = self.universal(h)                                 # U(x)  (teacher)
        d = self.task_experts[task](h)                        # D_n(x) (student)
        # Eq. 4 (with LoRA residual base from Eq. 1): h_t = W0 x + gu U(x) + gd D_n(x)
        h_final = h + g[..., 0:1] * u + g[..., 1:2] * d
        logits = self.head(h_final)                           # Eq. 5
        return logits, u, d, g

    # --- losses ---------------------------------------------------------------
    def gen_loss(self, ids, img, anchor_ids_b, task):
        """Lgen,n: next-token NLL (Eq. 7)."""
        logits, u, d, _ = self.forward(ids, img, anchor_ids_b, task)
        tgt = ids[:, 1:]
        logits = logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                               ignore_index=PAD)
        return loss, u, d


def info_nce(u, d, temp=0.1):
    """Knowledge harmonization (Eq. 6): maximise MI(U; D_n) via InfoNCE.

    Exact mutual information is intractable; InfoNCE is its standard tractable
    lower bound (van den Oord et al., 2018):
        I(U; D) >= log N - L_InfoNCE.
    Pseudocode of exact MI we approximate:
        I = E_{p(u,d)}[ log ( p(u,d) / (p(u) p(d)) ) ]
    We pool U(x) (teacher) and D_n(x) (student) per sample and pull positives
    together / push negatives apart in a contrastive batch.
    """
    u = F.normalize(u.mean(1), dim=-1)        # [B,D] teacher
    d = F.normalize(d.mean(1), dim=-1)        # [B,D] student
    logits = u @ d.t() / temp                 # [B,B]
    labels = torch.arange(u.size(0))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ---- downstream MIE backbone (the "plug-and-play" task model) ----------------
class MIEClassifier(nn.Module):
    """Tiny multimodal extractor: predicts the unified gold type. Stands in for
    the task backbones (PGIM/HVPNeT/UniCL...) that SAMA augments."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=PAD)
        self.img_enc = ImageEncoder()
        self.cls = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, len(GOLD2ID)))

    def forward(self, ids, img):
        t = self.emb(ids).mean(1)
        v = self.img_enc(img).mean(1)
        return self.cls(torch.cat([t, v], -1))
