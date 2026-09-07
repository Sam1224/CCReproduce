from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

PAD = 0
BOS = 1
EOS = 2


class ToySAMD2QDataset(Dataset):
    """Synthetic multimodal Doc2Query data for e-commerce expansion."""

    def __init__(self, num_samples: int = 1024, vocab_size: int = 256, title_len: int = 10, query_len: int = 7, image_dim: int = 64, seed: int = 11) -> None:
        self.rng = random.Random(seed)
        generator = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.title_len = title_len
        self.query_len = query_len
        self.image_dim = image_dim
        self.attribute_tokens = list(range(10, 40))
        self.samples: List[Dict[str, torch.Tensor]] = []
        image_attr_projection = torch.randn(vocab_size, image_dim, generator=generator)
        for _ in range(num_samples):
            title = torch.randint(40, vocab_size, (title_len,), generator=generator)
            visual_attrs = torch.tensor(self.rng.sample(self.attribute_tokens, 3), dtype=torch.long)
            target = torch.cat([torch.tensor([BOS]), visual_attrs[:2], title[: query_len - 4], torch.tensor([EOS])])[:query_len]
            if target.numel() < query_len:
                target = torch.cat([target, torch.full((query_len - target.numel(),), PAD)])
            masked_title = title.clone()
            masked_title[:2] = PAD
            image = image_attr_projection[visual_attrs].sum(dim=0) + torch.randn(image_dim, generator=generator) * 0.05
            demand = torch.rand((), generator=generator) * 0.7 + 0.3
            conversion = torch.rand((), generator=generator) * 0.5 + 0.2
            self.samples.append({
                "title": title,
                "masked_title": masked_title,
                "image": image.float(),
                "target_query": target.long(),
                "demand": demand.float(),
                "conversion": conversion.float(),
                "visual_attrs": visual_attrs,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
