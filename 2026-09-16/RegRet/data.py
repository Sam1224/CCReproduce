from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticRegRetConfig:
    region_dim: int = 32
    global_dim: int = 32
    text_dim: int = 32
    num_styles: int = 12
    num_contexts: int = 2
    num_attrs: int = 24
    num_backgrounds: int = 12
    noise_std: float = 0.18
    background_scale: float = 1.15
    region_scale_in_global: float = 0.25
    context_scale_in_region: float = 0.22
    context_scale_in_global: float = 0.85


def _make_prototypes(cfg: SyntheticRegRetConfig, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    style_base = torch.randn(cfg.num_styles, cfg.region_dim, generator=g)
    context_region = torch.randn(cfg.num_contexts, cfg.region_dim, generator=g)
    context_text = torch.randn(cfg.num_contexts, cfg.text_dim, generator=g)
    background_base = torch.randn(cfg.num_backgrounds, cfg.global_dim, generator=g)
    style_base = torch.nn.functional.normalize(style_base, dim=-1)
    context_region = torch.nn.functional.normalize(context_region, dim=-1)
    context_text = torch.nn.functional.normalize(context_text, dim=-1)
    background_base = torch.nn.functional.normalize(background_base, dim=-1)
    return style_base, context_region, context_text, background_base


class SyntheticRegRetDataset(Dataset):
    def __init__(self, n: int, cfg: SyntheticRegRetConfig, seed: int = 0):
        self.cfg = cfg
        self.n = n
        g = torch.Generator().manual_seed(seed)
        style_base, context_region, context_text, background_base = _make_prototypes(cfg)

        style_id = torch.randint(0, cfg.num_styles, (n,), generator=g)
        context_id = torch.randint(0, cfg.num_contexts, (n,), generator=g)
        attr_id = style_id * cfg.num_contexts + context_id
        background_id = torch.randint(0, cfg.num_backgrounds, (n,), generator=g)

        region_feat = (
            style_base[style_id]
            + cfg.context_scale_in_region * context_region[context_id]
            + cfg.noise_std * torch.randn(n, cfg.region_dim, generator=g)
        )
        text_feat = (
            style_base[style_id][:, : cfg.text_dim]
            + 0.75 * context_text[context_id]
            + (cfg.noise_std * 0.5) * torch.randn(n, cfg.text_dim, generator=g)
        )
        global_feat = (
            cfg.background_scale * background_base[background_id]
            + cfg.context_scale_in_global * context_region[context_id][:, : cfg.global_dim]
            + cfg.region_scale_in_global * style_base[style_id][:, : cfg.global_dim]
            + cfg.noise_std * torch.randn(n, cfg.global_dim, generator=g)
        )

        self.region_feat = torch.nn.functional.normalize(region_feat, dim=-1)
        self.global_feat = torch.nn.functional.normalize(global_feat, dim=-1)
        self.text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        self.attr_id = attr_id
        self.background_id = background_id

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "region_feat": self.region_feat[idx],
            "global_feat": self.global_feat[idx],
            "text_feat": self.text_feat[idx],
            "attr_id": self.attr_id[idx],
            "background_id": self.background_id[idx],
        }
