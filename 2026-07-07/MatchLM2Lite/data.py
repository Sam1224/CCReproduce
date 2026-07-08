"""
MatchLM2Lite — Data utilities

Reproduced Content Identification (RCI) dataset:
- Input: pairs of videos (reference, candidate)
- Output: fine-grained reproduction score (0-1)

Multimodal signals:
- Video: frame-level visual features
- Audio: acoustic features
- Text: title, description, captions/subtitles
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class VideoMeta:
    """Metadata for a single video."""
    video_id: str
    title: str
    description: str
    captions: list[str] = field(default_factory=list)    # auto-generated captions
    duration_seconds: float = 60.0


@dataclass
class VideoPair:
    """A pair of videos with a reproduction label."""
    reference: VideoMeta    # the original video
    candidate: VideoMeta    # potentially reproduced video
    label: float            # 0.0 = distinct, 1.0 = reproduced
    reproduction_type: str = ""  # e.g. "clip_repost", "dubbed", "caption_replaced", "reframed"


# Toy video pairs simulating e-commerce content platform data
_TOY_PAIRS = [
    VideoPair(
        reference=VideoMeta("vid_001", "达人开箱视频：限量版球鞋评测",
                            "博主亲测 Jordan 1 Retro 开箱体验", ["耐克 乔丹 球鞋 开箱"]),
        candidate=VideoMeta("vid_002", "球鞋开箱测评——Jordan 1 限定款",
                            "up主体验 Jordan 1 限定版真实评测", ["Jordan 球鞋 测评 开箱"]),
        label=0.95,
        reproduction_type="repost_minor_edit",
    ),
    VideoPair(
        reference=VideoMeta("vid_003", "小姐姐展示新款汉服穿搭",
                            "传统汉服穿搭教学，适合日常佩戴", ["汉服 穿搭 展示"]),
        candidate=VideoMeta("vid_004", "2026最新汉服穿搭推荐",
                            "达人推荐汉服穿搭，时尚传统结合", ["汉服 传统服饰 推荐"]),
        label=0.0,
        reproduction_type="",
    ),
    VideoPair(
        reference=VideoMeta("vid_005", "化妆品测评：新品口红试色",
                            "博主亲测 5 款口红，详细试色", ["口红 试色 美妆"]),
        candidate=VideoMeta("vid_006", "口红试色分享——配音版",
                            "原视频配上新解说（搬运+配音替换）", ["口红 美妆 评测"]),
        label=0.88,
        reproduction_type="dubbed",
    ),
    VideoPair(
        reference=VideoMeta("vid_007", "网红零食盲盒测评",
                            "购买网红零食盲盒，逐一品尝测评", ["零食 盲盒 测评"]),
        candidate=VideoMeta("vid_008", "美食测评频道精选",
                            "多档美食节目合集，独立内容", ["美食 测评 合集"]),
        label=0.0,
        reproduction_type="",
    ),
    VideoPair(
        reference=VideoMeta("vid_009", "直播间爆款商品推荐",
                            "达人直播推荐今日爆款商品", ["直播 爆款 推荐"]),
        candidate=VideoMeta("vid_010", "爆款商品推荐剪辑版",
                            "截取原直播精彩片段，字幕替换", ["商品 推荐 剪辑"]),
        label=0.92,
        reproduction_type="clip_repost",
    ),
    VideoPair(
        reference=VideoMeta("vid_011", "健身教程：腹肌训练计划",
                            "专业健身教练分享腹肌训练方法", ["健身 腹肌 训练"]),
        candidate=VideoMeta("vid_012", "家庭健身方案——AI 生成字幕版",
                            "同视频内容，仅替换字幕标注", ["健身 家庭 训练"]),
        label=0.85,
        reproduction_type="caption_replaced",
    ),
]


class RCIDataset(Dataset):
    """Reproduced Content Identification dataset.

    In production (TikTok): millions of video pairs annotated by
    manual reviewers + model-assisted labeling.
    """

    def __init__(
        self,
        pairs: list[VideoPair] | None = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.pairs = pairs or list(_TOY_PAIRS)
        if augment:
            # Augment by adding pairs with reversed reference/candidate
            augmented = []
            for p in self.pairs:
                if p.label > 0.5:  # only flip actual reproductions
                    augmented.append(VideoPair(p.candidate, p.reference, p.label, p.reproduction_type))
            self.pairs = self.pairs + augmented

        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            "ref_title": pair.reference.title,
            "ref_description": pair.reference.description,
            "ref_captions": " ".join(pair.reference.captions),
            "cand_title": pair.candidate.title,
            "cand_description": pair.candidate.description,
            "cand_captions": " ".join(pair.candidate.captions),
            "label": pair.label,
            "reproduction_type": pair.reproduction_type,
        }


# ─── Feature Simulation ───────────────────────────────────────────────────────

def simulate_visual_features(batch_size: int, num_frames: int = 8,
                              feat_dim: int = 512) -> torch.Tensor:
    """Simulate pre-extracted frame-level visual features.

    In production: extracted by ViT/CLIP from sampled video frames.
    """
    return torch.randn(batch_size, num_frames, feat_dim)


def simulate_audio_features(batch_size: int, num_segments: int = 4,
                             feat_dim: int = 256) -> torch.Tensor:
    """Simulate pre-extracted audio features.

    In production: extracted by audio encoder (e.g., wav2vec2) from audio track.
    """
    return torch.randn(batch_size, num_segments, feat_dim)
