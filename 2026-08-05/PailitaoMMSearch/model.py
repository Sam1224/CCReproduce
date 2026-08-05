"""model.py

最小可运行的 Pailitao-MMSearch toy 复现核心组件。

包含三块：
1) HybSID 双塔 Encoder：连续表示 + 离散语义 ID（向量量化）
2) 两阶段持续预训练/蒸馏：这里不写训练 loop，但提供 loss/teacher 等接口
3) 混合推理后训练：提供可训练的 HybridScoreHead，用于把“离散匹配 + 连续相似度”融合到最终分数

与论文真实实现的差距：
- 真实系统可能采用更复杂 backbone（ViT/Transformer）、更复杂的离散语义 tokenizer、多粒度 ID、以及大规模检索工程。
- 本实现用 group VQ 近似离散语义 ID，并用一个 toy teacher（固定随机特征）演示蒸馏接口。

(Pseudo) 更贴近论文但未实现的训练/系统：

```text
for stage in [continual_pretrain, distill, posttrain_hybrid_inference]:
    for batch in dataloader:
        zq_cont, sid_q, zq_quant, zq_hyb = encoder(query)
        zi_cont, sid_i, zi_quant, zi_hyb = encoder(item)

        # multi-task losses (not fully implemented)
        loss = nce(zq_hyb, zi_hyb)
        loss += vq_commit_loss
        if stage == distill:
            loss += mse(zq_cont, teacher(query)) + mse(zi_cont, teacher(item))
        if stage == posttrain:
            loss = rank_loss(score_head(zq, zi), score_head(zq, neg))
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class SimpleTextEncoder(nn.Module):
    """极简文本 encoder：Embedding + mean pooling。"""

    def __init__(self, vocab_size: int, d_text: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_text)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        x = self.emb(input_ids)  # [B, L, d]
        feat = x.mean(dim=1)
        return feat


class SimpleImageEncoder(nn.Module):
    """极简图像 encoder：小 CNN -> 向量。"""

    def __init__(self, d_img: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, d_img),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, 3, 32, 32]
        return self.net(image)


class GroupVectorQuantizer(nn.Module):
    """Group Vector Quantization，用于模拟 HybSID 的“离散语义 ID”。

    - 将输入向量切成 num_groups 个子向量
    - 每组在 codebook 中做最近邻量化，输出：
      - sids: [B, G]（离散语义 ID）
      - z_q: [B, D]（量化后的连续向量，使用 straight-through estimator 反传）
      - vq_loss: commitment loss（可选）

    注意：真实论文系统的离散 ID 可能来自更复杂的 tokenizer/训练策略；这里用 VQ 近似。
    """

    def __init__(self, dim: int, num_groups: int = 4, num_codes: int = 64, beta: float = 0.25):
        super().__init__()
        assert dim % num_groups == 0, "dim 必须能被 num_groups 整除"
        self.dim = dim
        self.num_groups = num_groups
        self.num_codes = num_codes
        self.beta = beta
        self.group_dim = dim // num_groups

        self.codebook = nn.Embedding(num_groups * num_codes, self.group_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, D]
        B, D = z.shape
        z_g = z.view(B, self.num_groups, self.group_dim)  # [B, G, d]

        # codebook reshape: [G, K, d]
        cb = self.codebook.weight.view(self.num_groups, self.num_codes, self.group_dim)

        # 计算距离并取 argmin
        # dist: [B, G, K]
        dist = (
            (z_g.unsqueeze(2) - cb.unsqueeze(0)) ** 2
        ).sum(dim=-1)
        sids = dist.argmin(dim=-1)  # [B, G]

        # gather code vectors
        # idx: [B, G] -> [B, G, 1]
        idx = sids.unsqueeze(-1)
        cb_expand = cb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, G, K, d]
        z_q = torch.gather(cb_expand, dim=2, index=idx.unsqueeze(-1).expand(-1, -1, 1, self.group_dim)).squeeze(2)

        # straight-through estimator
        z_q_st = z_g + (z_q - z_g).detach()

        # commitment loss
        vq_loss = F.mse_loss(z_q.detach(), z_g) + self.beta * F.mse_loss(z_q, z_g.detach())

        return sids, z_q_st.view(B, D), vq_loss


class HybridScoreHead(nn.Module):
    """混合推理打分头（用于 stage3 post-train）。

    给定：
    - 连续相似度 s_cont（cosine）
    - 离散相似度 s_sid（加权匹配率）

    输出最终分数：
        s = w_cont * s_cont + w_sid * s_sid

    训练时可以冻结 encoder，仅训练本 head，使推理更贴近“离散过滤 + 连续 rerank”的形态。
    """

    def __init__(self, num_groups: int):
        super().__init__()
        self.w_cont = nn.Parameter(torch.tensor(1.0))
        self.w_sid = nn.Parameter(torch.tensor(1.0))
        self.sid_group_weight = nn.Parameter(torch.ones(num_groups))

    def sid_similarity(self, sid_q: torch.Tensor, sid_i: torch.Tensor) -> torch.Tensor:
        # sid_q/sid_i: [B, G]
        match = (sid_q == sid_i).float()  # [B, G]
        w = F.softplus(self.sid_group_weight)  # >0
        s = (match * w.unsqueeze(0)).sum(dim=-1) / (w.sum() + 1e-8)
        return s

    def forward(
        self,
        zq_cont: torch.Tensor,
        zi_cont: torch.Tensor,
        sid_q: torch.Tensor,
        sid_i: torch.Tensor,
    ) -> torch.Tensor:
        s_cont = (l2_normalize(zq_cont) * l2_normalize(zi_cont)).sum(dim=-1)  # [B]
        s_sid = self.sid_similarity(sid_q, sid_i)
        return self.w_cont * s_cont + self.w_sid * s_sid


@dataclass
class HybSIDOutputs:
    z_cont: torch.Tensor   # [B, D]
    sid: torch.Tensor      # [B, G]
    z_quant: torch.Tensor  # [B, D]
    z_hyb: torch.Tensor    # [B, D]
    vq_loss: torch.Tensor  # scalar


class HybSIDDualEncoder(nn.Module):
    """HybSID 双塔（query/item 共用 encoder）。

    输出同时包含：
    - 连续向量 z_cont
    - 离散语义 ID sid
    - 量化向量 z_quant
    - 混合向量 z_hyb（用 gate 在 z_cont 与 z_quant 间插值）

    注意：
    - 真实论文系统的 HybSID 可能在 ID 生成、融合方式、token 粒度等方面更复杂。
    - 本 toy 用 VQ+线性 gate 仅保留关键“形状”。
    """

    def __init__(
        self,
        vocab_size: int,
        d_text: int = 128,
        d_img: int = 128,
        d_model: int = 128,
        num_groups: int = 4,
        num_codes: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        assert d_model % num_groups == 0
        self.temperature = temperature

        self.text_enc = SimpleTextEncoder(vocab_size=vocab_size, d_text=d_text)
        self.img_enc = SimpleImageEncoder(d_img=d_img)

        self.fuse = nn.Sequential(
            nn.Linear(d_text + d_img, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        self.vq = GroupVectorQuantizer(dim=d_model, num_groups=num_groups, num_codes=num_codes)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid -> lam

        self.score_head = HybridScoreHead(num_groups=num_groups)

    @property
    def num_groups(self) -> int:
        return self.vq.num_groups

    def encode(self, input_ids: torch.Tensor, image: torch.Tensor) -> HybSIDOutputs:
        t = self.text_enc(input_ids)
        v = self.img_enc(image)
        z = self.fuse(torch.cat([t, v], dim=-1))

        sid, z_quant, vq_loss = self.vq(z)
        z_cont = z

        lam = torch.sigmoid(self.gate_logit)
        z_hyb = lam * z_cont + (1.0 - lam) * z_quant

        return HybSIDOutputs(
            z_cont=l2_normalize(z_cont),
            sid=sid,
            z_quant=l2_normalize(z_quant),
            z_hyb=l2_normalize(z_hyb),
            vq_loss=vq_loss,
        )

    def contrastive_loss(self, zq: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        """双向 in-batch InfoNCE。"""
        # zq/zi: [B, D]
        logits = (zq @ zi.t()) / self.temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_q = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_q + loss_i)


class ToyTeacherEncoder(nn.Module):
    """用于 stage2 蒸馏的 toy teacher。

    设计：
    - 用固定随机 embedding + 线性层生成表征（不训练，模拟一个“更稳定/更强的 teacher”信号）

    这并不是论文真实 teacher 的实现，仅用于演示蒸馏接口。
    """

    def __init__(self, vocab_size: int, d_out: int = 128, seed: int = 1234):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        self.word_emb = nn.Embedding(vocab_size, 64)
        self.img_proj = nn.Linear(3, 64)
        self.out = nn.Linear(128, d_out)

        # 固定随机参数
        with torch.no_grad():
            self.word_emb.weight.copy_(torch.randn_like(self.word_emb.weight, generator=g) * 0.2)
            for p in self.img_proj.parameters():
                p.copy_(torch.randn_like(p, generator=g) * 0.2)
            for p in self.out.parameters():
                p.copy_(torch.randn_like(p, generator=g) * 0.2)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # text: mean pooled
        t = self.word_emb(input_ids).mean(dim=1)  # [B, 64]
        # image: global mean RGB
        rgb = image.mean(dim=[2, 3])  # [B, 3]
        v = torch.tanh(self.img_proj(rgb))  # [B, 64]
        z = self.out(torch.cat([t, v], dim=-1))
        return l2_normalize(z)


def margin_ranking_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """简单的 pairwise hinge ranking loss。

    pos_score: [B]
    neg_score: [B]
    """
    return torch.relu(margin - pos_score + neg_score).mean()
