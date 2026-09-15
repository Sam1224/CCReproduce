from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class ViduS2Sample:
    """单条样本的结构化描述（主要用于 type hint / 文档）。"""

    avatar_code: torch.Tensor  # (A,)
    ref_attr_ids: torch.Tensor  # (3,) -> (bg, clothes, style)
    ref_update_attr_ids: torch.Tensor  # (3,)
    ref_update_t: int
    stream_attr_ids: torch.Tensor  # (T, 3)
    edit_flags: torch.Tensor  # (T, 3) float32 in {0,1}, 对应 (bg, clothes, style)
    target_attr_ids: torch.Tensor  # (T, 3)
    target_frames: torch.Tensor  # (T, 3, H, W)


class AttributeCodec:
    """把离散属性 <-> 16x16 合成帧互相编码/解码。

    约定：
    - background：占据大部分区域
    - clothes：中部矩形
    - face：上部矩形（由 avatar_code 决定颜色，提供 identity 线索）
    - style：左上角 3x3 patch（便于可解释评测）
    """

    def __init__(self, h: int = 16, w: int = 16) -> None:
        self.h = int(h)
        self.w = int(w)

        # (num_ids, 3)
        self.bg_palette = torch.tensor(
            [
                [0.10, 0.10, 0.60],
                [0.10, 0.60, 0.10],
                [0.60, 0.10, 0.10],
            ],
            dtype=torch.float32,
        )
        self.clothes_palette = torch.tensor(
            [
                [0.90, 0.80, 0.10],
                [0.10, 0.80, 0.90],
                [0.90, 0.10, 0.80],
            ],
            dtype=torch.float32,
        )
        self.style_palette = torch.tensor(
            [
                [0.95, 0.95, 0.95],
                [0.30, 0.30, 0.30],
                [0.70, 0.95, 0.70],
            ],
            dtype=torch.float32,
        )

        # 空间区域（用 mask 便于 batched mean）
        self._mask_style = torch.zeros(1, 1, self.h, self.w, dtype=torch.float32)
        self._mask_style[:, :, :3, :3] = 1.0

        self._mask_face = torch.zeros(1, 1, self.h, self.w, dtype=torch.float32)
        self._mask_face[:, :, 2:6, 6:10] = 1.0

        self._mask_clothes = torch.zeros(1, 1, self.h, self.w, dtype=torch.float32)
        self._mask_clothes[:, :, 6:13, 5:11] = 1.0

        self._mask_bg = torch.ones(1, 1, self.h, self.w, dtype=torch.float32)
        self._mask_bg = self._mask_bg * (1.0 - self._mask_style)
        self._mask_bg = self._mask_bg * (1.0 - self._mask_face)
        self._mask_bg = self._mask_bg * (1.0 - self._mask_clothes)

        self._eps = 1e-6

    @property
    def num_bg(self) -> int:
        return int(self.bg_palette.shape[0])

    @property
    def num_clothes(self) -> int:
        return int(self.clothes_palette.shape[0])

    @property
    def num_style(self) -> int:
        return int(self.style_palette.shape[0])

    def avatar_to_face_color(self, avatar_code: torch.Tensor) -> torch.Tensor:
        """(A,) -> (3,)"""
        x = torch.sigmoid(avatar_code[:3])
        return 0.35 + 0.55 * x

    def render_frame(
        self,
        avatar_code: torch.Tensor,
        bg_id: int,
        clothes_id: int,
        style_id: int,
    ) -> torch.Tensor:
        """返回 (3, H, W) float32 in [0,1]."""
        bg = self.bg_palette[int(bg_id)]
        clothes = self.clothes_palette[int(clothes_id)]
        style = self.style_palette[int(style_id)]
        face = self.avatar_to_face_color(avatar_code)

        frame = torch.zeros(3, self.h, self.w, dtype=torch.float32)
        frame[:] = bg[:, None, None]

        # overwrite regions
        frame[:, 6:13, 5:11] = clothes[:, None, None]
        frame[:, 2:6, 6:10] = face[:, None, None]
        frame[:, :3, :3] = style[:, None, None]
        return frame

    def _region_mean(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """frames: (..., 3, H, W), mask: (1,1,H,W) -> (..., 3)"""
        while mask.dim() < frames.dim():
            mask = mask.unsqueeze(0)
        denom = mask.sum(dim=(-2, -1)).clamp_min(self._eps)
        num = (frames * mask).sum(dim=(-2, -1))
        return num / denom

    def estimate_attr_ids(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) -> (B, T, 3) long

        用颜色最近邻来估计 (bg, clothes, style) id。
        """
        bg_mean = self._region_mean(frames, self._mask_bg)
        clothes_mean = self._region_mean(frames, self._mask_clothes)
        style_mean = self._region_mean(frames, self._mask_style)

        def nn_id(x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
            # x: (..., 3), palette: (K, 3)
            diff = x.unsqueeze(-2) - palette.to(x.device).unsqueeze(0)
            dist = (diff * diff).sum(dim=-1)
            return dist.argmin(dim=-1)

        bg_id = nn_id(bg_mean, self.bg_palette)
        clothes_id = nn_id(clothes_mean, self.clothes_palette)
        style_id = nn_id(style_mean, self.style_palette)
        return torch.stack([bg_id, clothes_id, style_id], dim=-1).long()


def _randint(g: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low, high, (1,), generator=g).item())


def _randint_different(g: torch.Generator, high: int, avoid: int) -> int:
    if high <= 1:
        return 0
    x = _randint(g, 0, high)
    while x == int(avoid):
        x = _randint(g, 0, high)
    return x


class ToyViduS2Dataset(Dataset):
    """合成视频序列数据。

    - stream_attr_ids: 类似实时交互中的 prompt/state
    - ref_attr_ids: 类似参考帧/参考图像描述
    - edit_flags: 指定每个属性是否从 reference 复制（editing）
    - ref_update_t + ref_update_attr_ids: 流式过程中 reference 更新
    """

    def __init__(
        self,
        length: int = 256,
        t: int = 12,
        h: int = 16,
        w: int = 16,
        avatar_dim: int = 8,
        enable_edits: bool = True,
        enable_ref_update: bool = True,
        force_edit_clothes_style: bool = True,
        seed: int = 0,
    ) -> None:
        self.length = int(length)
        self.t = int(t)
        self.codec = AttributeCodec(h=h, w=w)
        self.avatar_dim = int(avatar_dim)
        self.enable_edits = bool(enable_edits)
        self.enable_ref_update = bool(enable_ref_update)
        self.force_edit_clothes_style = bool(force_edit_clothes_style)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))

        avatar_code = torch.randn(self.avatar_dim, generator=g)

        # stream attrs（模拟实时 prompt/state）
        stream_bg = _randint(g, 0, self.codec.num_bg)
        stream_clothes = _randint(g, 0, self.codec.num_clothes)
        stream_style = _randint(g, 0, self.codec.num_style)

        stream_attr_ids = torch.tensor(
            [[stream_bg, stream_clothes, stream_style] for _ in range(self.t)],
            dtype=torch.long,
        )

        # reference attrs 1/2，保证与 stream 在 clothes/style 上不同，才能在 test 中清晰对比 baseline
        ref_bg = _randint(g, 0, self.codec.num_bg)
        ref_clothes = _randint_different(g, self.codec.num_clothes, stream_clothes)
        ref_style = _randint_different(g, self.codec.num_style, stream_style)

        ref2_bg = _randint(g, 0, self.codec.num_bg)
        ref2_clothes = _randint_different(g, self.codec.num_clothes, ref_clothes)
        # 进一步保证 ref2 与 stream 在 clothes/style 上也不同，便于 baseline vs editing 的对比更清晰
        if ref2_clothes == int(stream_clothes):
            ref2_clothes = _randint_different(g, self.codec.num_clothes, stream_clothes)

        ref2_style = _randint_different(g, self.codec.num_style, ref_style)
        if ref2_style == int(stream_style):
            ref2_style = _randint_different(g, self.codec.num_style, stream_style)

        ref_attr_ids = torch.tensor([ref_bg, ref_clothes, ref_style], dtype=torch.long)
        ref_update_attr_ids = torch.tensor([ref2_bg, ref2_clothes, ref2_style], dtype=torch.long)

        if self.enable_ref_update:
            ref_update_t = _randint(g, self.t // 3, 2 * self.t // 3)
        else:
            ref_update_t = self.t + 1

        # edit flags: (T, 3)
        edit_flags = torch.zeros(self.t, 3, dtype=torch.float32)
        if self.enable_edits:
            if self.force_edit_clothes_style:
                edit_flags[:, 1] = 1.0
                edit_flags[:, 2] = 1.0
                # background 默认不 edit；但保留接口
                edit_flags[:, 0] = 0.0
            else:
                # 更随机的 editing（保证至少两类出现）
                edit_flags[:, 0] = (torch.rand(self.t, generator=g) < 0.2).float()
                edit_flags[:, 1] = (torch.rand(self.t, generator=g) < 0.8).float()
                edit_flags[:, 2] = (torch.rand(self.t, generator=g) < 0.8).float()

        # target attrs: 根据 edit_flags 从 stream/ref 中混合
        target_attr_ids = torch.empty_like(stream_attr_ids)
        for ti in range(self.t):
            cur_ref = ref_attr_ids if ti < ref_update_t else ref_update_attr_ids
            for k in range(3):
                if float(edit_flags[ti, k].item()) > 0.5:
                    target_attr_ids[ti, k] = cur_ref[k]
                else:
                    target_attr_ids[ti, k] = stream_attr_ids[ti, k]

        # render target frames
        target_frames = []
        for ti in range(self.t):
            bg_id, clothes_id, style_id = [int(x.item()) for x in target_attr_ids[ti]]
            target_frames.append(self.codec.render_frame(avatar_code, bg_id, clothes_id, style_id))
        target_frames = torch.stack(target_frames, dim=0)  # (T,3,H,W)

        # reference frames（仅用于“reference-conditioned”语义完整；模型内部主要用 ref_attr_ids）
        ref_frame = self.codec.render_frame(
            avatar_code,
            int(ref_attr_ids[0].item()),
            int(ref_attr_ids[1].item()),
            int(ref_attr_ids[2].item()),
        )
        ref_update_frame = self.codec.render_frame(
            avatar_code,
            int(ref_update_attr_ids[0].item()),
            int(ref_update_attr_ids[1].item()),
            int(ref_update_attr_ids[2].item()),
        )

        return {
            "avatar_code": avatar_code,
            "ref_attr_ids": ref_attr_ids,
            "ref_frame": ref_frame,
            "ref_update_attr_ids": ref_update_attr_ids,
            "ref_update_frame": ref_update_frame,
            "ref_update_t": torch.tensor(ref_update_t, dtype=torch.long),
            "stream_attr_ids": stream_attr_ids,
            "edit_flags": edit_flags,
            "target_attr_ids": target_attr_ids,
            "target_frames": target_frames,
        }
