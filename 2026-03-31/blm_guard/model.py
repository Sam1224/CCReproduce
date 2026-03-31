from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()


def aks_bin_top(scores: torch.Tensor, m: int = 3) -> torch.Tensor:
    """BIN+TOP selection.

    scores: (N,) higher means more risky.
    """

    N = scores.shape[0]
    m = min(m, N)
    bins = m
    idx = []
    for b in range(bins):
        lo = int(math.floor(b * N / bins))
        hi = int(math.floor((b + 1) * N / bins))
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

    return torch.tensor(idx[:m], dtype=torch.long)


def select_regions(frames: torch.Tensor, top_patches: int = 4) -> torch.Tensor:
    """Toy patch selection by L2 norm.

    frames: (m, d). We simulate patches by splitting dims.
    returns: pooled salient region features (m, d)
    """

    m, d = frames.shape
    patches = frames.view(m, top_patches, d // top_patches)
    sal = patches.norm(dim=-1)  # (m, P)
    top = torch.argmax(sal, dim=-1)
    out = patches[torch.arange(m), top]
    # expand back to d by repeating
    return out.repeat_interleave(top_patches, dim=-1)


class TinyText(nn.Module):
    def __init__(self, vocab: int, d: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids).mean(dim=0)


class BLMGuardModel(nn.Module):
    def __init__(self, d_in: int = 64, d: int = 128, vocab: int = 800, n_scene: int = 7, n_type: int = 5) -> None:
        super().__init__()
        self.asr = TinyText(vocab, d)
        self.frame = nn.Linear(d_in, d)

        self.backbone = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
        )

        self.scene = nn.Linear(d, n_scene)
        self.type_head = nn.Linear(d, n_type)
        self.risky = nn.Linear(d, 1)

    def forward(self, frame_feat: torch.Tensor, asr_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.asr(asr_ids)
        f = self.frame(frame_feat)
        h = self.backbone(torch.cat([f, a], dim=-1))
        return self.scene(h), self.type_head(h), self.risky(h).squeeze(-1)


def hybrid_reward(scene_pred: int, type_pred: int, scene_gt: int, type_gt: int) -> float:
    # rrule
    if scene_pred == scene_gt and type_pred == type_gt:
        r_rule = 1.0
    elif scene_pred == scene_gt or type_pred == type_gt:
        r_rule = 0.5
    else:
        r_rule = 0.0

    # rformat: always produce <think>/<answer> in this toy
    r_format = 1.0

    # rscaR: self-consistency proxy: require the label is non-null when risky
    r_scar = 1.0 if (scene_pred >= 0 and type_pred >= 0) else 0.0

    return r_rule + 0.1 * r_format + 0.4 * r_scar


def group_advantage(rewards: torch.Tensor) -> torch.Tensor:
    mu = rewards.mean()
    std = rewards.std().clamp_min(1e-6)
    return (rewards - mu) / std


def logprob_from_logits(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
