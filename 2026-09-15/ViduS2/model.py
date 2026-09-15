from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StreamState:
    """模拟论文中的 streaming state。

    - h: avatar/stream hidden state（实时生成时的缓存状态）
    - ref_attr_ids: 当前 reference 的离散属性（bg/clothes/style）
    - avatar_color: 由 avatar_code 得到的 identity 颜色（在 face 区域保持一致）
    """

    h: torch.Tensor  # (B, hidden)
    ref_attr_ids: torch.Tensor  # (B, 3)
    avatar_color: torch.Tensor  # (B, 3)


class StreamingViduS2Model(nn.Module):
    """toy streaming video generator。

    与其直接回归整张图像像素，本 toy 使用一个**可解释的空间解码器**：

    - 模型每步预测 3 个区域颜色（bg/clothes/style），再通过固定 mask 合成帧。
    - 好处：训练更稳定；同时更贴近论文里“空间一致 / 可编辑属性”的抽象。

    设计点：
    - step() 是单步实时推理接口
    - update_reference() 可在流式过程中修改 reference（mid-stream update）
    - edit_flags 决定每个属性来自 stream 还是 reference（reference-conditioned editing）
    """

    def __init__(
        self,
        num_bg: int = 3,
        num_clothes: int = 3,
        num_style: int = 3,
        avatar_dim: int = 8,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        frame_shape: Tuple[int, int, int] = (3, 16, 16),
        allow_reference: bool = True,
    ) -> None:
        super().__init__()
        self.allow_reference = bool(allow_reference)

        c, h, w = [int(x) for x in frame_shape]
        self.frame_shape = (c, h, w)

        self.bg_emb = nn.Embedding(num_bg, emb_dim)
        self.clothes_emb = nn.Embedding(num_clothes, emb_dim)
        self.style_emb = nn.Embedding(num_style, emb_dim)

        self.avatar_proj = nn.Linear(avatar_dim, hidden_dim)
        self.avatar_to_color = nn.Linear(avatar_dim, 3)

        self.rnn = nn.GRUCell(input_size=emb_dim * 3 + 3, hidden_size=hidden_dim)

        # 固定 palette（与 data.py 的 AttributeCodec 对齐）。
        # 这样 toy 实验的差异主要来自“是否具备 reference editing”，而不是 palette 学习是否收敛。
        self.register_buffer(
            "bg_palette",
            torch.tensor(
                [
                    [0.10, 0.10, 0.60],
                    [0.10, 0.60, 0.10],
                    [0.60, 0.10, 0.10],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "clothes_palette",
            torch.tensor(
                [
                    [0.90, 0.80, 0.10],
                    [0.10, 0.80, 0.90],
                    [0.90, 0.10, 0.80],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "style_palette",
            torch.tensor(
                [
                    [0.95, 0.95, 0.95],
                    [0.30, 0.30, 0.30],
                    [0.70, 0.95, 0.70],
                ],
                dtype=torch.float32,
            ),
        )

        # 固定空间 mask（与 data.py 的 AttributeCodec 对齐）
        mask_style = torch.zeros(1, 1, h, w)
        mask_style[:, :, :3, :3] = 1.0

        mask_face = torch.zeros(1, 1, h, w)
        mask_face[:, :, 2:6, 6:10] = 1.0

        mask_clothes = torch.zeros(1, 1, h, w)
        mask_clothes[:, :, 6:13, 5:11] = 1.0

        mask_bg = torch.ones(1, 1, h, w)
        mask_bg = mask_bg * (1.0 - mask_style)
        mask_bg = mask_bg * (1.0 - mask_face)
        mask_bg = mask_bg * (1.0 - mask_clothes)

        self.register_buffer("mask_bg", mask_bg)
        self.register_buffer("mask_clothes", mask_clothes)
        self.register_buffer("mask_style", mask_style)
        self.register_buffer("mask_face", mask_face)

    def reset_stream(self, avatar_code: torch.Tensor) -> StreamState:
        """avatar_code: (B, A)"""
        h0 = torch.tanh(self.avatar_proj(avatar_code))
        avatar_color = torch.sigmoid(self.avatar_to_color(avatar_code))
        b = int(avatar_code.shape[0])
        ref_attr_ids = torch.zeros(b, 3, device=avatar_code.device, dtype=torch.long)
        return StreamState(h=h0, ref_attr_ids=ref_attr_ids, avatar_color=avatar_color)

    @torch.no_grad()
    def update_reference(
        self,
        state: StreamState,
        ref_attr_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> StreamState:
        """更新 reference。

        ref_attr_ids: (B,3)
        mask: (B,) bool，表示 batch 内哪些样本需要更新
        """
        if not self.allow_reference:
            return state
        if mask is None:
            state.ref_attr_ids = ref_attr_ids
            return state
        mask = mask.to(dtype=torch.bool, device=ref_attr_ids.device)
        state.ref_attr_ids = torch.where(mask[:, None], ref_attr_ids, state.ref_attr_ids)
        return state

    def _compose_frame(
        self,
        bg_color: torch.Tensor,
        clothes_color: torch.Tensor,
        style_color: torch.Tensor,
        avatar_color: torch.Tensor,
    ) -> torch.Tensor:
        """颜色 + mask 合成。

        inputs: (B,3)
        return: (B,3,H,W)
        """
        bg = bg_color[:, :, None, None] * self.mask_bg
        clothes = clothes_color[:, :, None, None] * self.mask_clothes
        style = style_color[:, :, None, None] * self.mask_style
        face = avatar_color[:, :, None, None] * self.mask_face
        return bg + clothes + style + face

    def step(
        self,
        state: StreamState,
        stream_attr_ids: torch.Tensor,
        edit_flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, StreamState]:
        """单步生成。"""
        if not self.allow_reference:
            edit_flags = torch.zeros_like(edit_flags)

        bg_id = stream_attr_ids[:, 0]
        clothes_id = stream_attr_ids[:, 1]
        style_id = stream_attr_ids[:, 2]

        ref_bg_id = state.ref_attr_ids[:, 0]
        ref_clothes_id = state.ref_attr_ids[:, 1]
        ref_style_id = state.ref_attr_ids[:, 2]

        bg_e = self.bg_emb(bg_id)
        clothes_e = self.clothes_emb(clothes_id)
        style_e = self.style_emb(style_id)

        ref_bg_e = self.bg_emb(ref_bg_id)
        ref_clothes_e = self.clothes_emb(ref_clothes_id)
        ref_style_e = self.style_emb(ref_style_id)

        # reference-conditioned editing：embedding space mix
        eb = bg_e * (1.0 - edit_flags[:, 0:1]) + ref_bg_e * edit_flags[:, 0:1]
        ec = clothes_e * (1.0 - edit_flags[:, 1:2]) + ref_clothes_e * edit_flags[:, 1:2]
        es = style_e * (1.0 - edit_flags[:, 2:3]) + ref_style_e * edit_flags[:, 2:3]

        cond = torch.cat([eb, ec, es, edit_flags.to(dtype=eb.dtype)], dim=-1)
        state.h = self.rnn(cond, state.h)

        # 用 edit_flags 在 id space 选择 effective attribute（更符合“把属性拷贝到 reference”的直觉）
        eff_bg_id = torch.where(edit_flags[:, 0] > 0.5, ref_bg_id, bg_id)
        eff_clothes_id = torch.where(edit_flags[:, 1] > 0.5, ref_clothes_id, clothes_id)
        eff_style_id = torch.where(edit_flags[:, 2] > 0.5, ref_style_id, style_id)

        bg_color = self.bg_palette[eff_bg_id]
        clothes_color = self.clothes_palette[eff_clothes_id]
        style_color = self.style_palette[eff_style_id]

        frame = self._compose_frame(bg_color, clothes_color, style_color, state.avatar_color)
        return frame, state

    def rollout(
        self,
        avatar_code: torch.Tensor,
        ref_attr_ids: torch.Tensor,
        ref_update_attr_ids: torch.Tensor,
        ref_update_t: torch.Tensor,
        stream_attr_ids: torch.Tensor,
        edit_flags: torch.Tensor,
    ) -> torch.Tensor:
        """流式 rollout: (B,T,3,H,W)"""
        b, t, _ = stream_attr_ids.shape

        state = self.reset_stream(avatar_code)
        state = self.update_reference(state, ref_attr_ids)

        frames = []
        for ti in range(t):
            mask = ref_update_t == int(ti)
            if mask.any():
                state = self.update_reference(state, ref_update_attr_ids, mask=mask)
            frame, state = self.step(state, stream_attr_ids[:, ti], edit_flags[:, ti])
            frames.append(frame)
        return torch.stack(frames, dim=1)
