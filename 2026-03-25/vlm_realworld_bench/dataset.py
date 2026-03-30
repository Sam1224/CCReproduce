from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


QUESTIONS = ["left_of", "right_of", "closest", "count"]


@dataclass
class Batch:
    img: torch.Tensor  # (N, D)
    q: torch.Tensor  # (N,)
    y: torch.Tensor  # (N,)


def _make_scene(g: torch.Generator, num_obj: int = 6, ood: bool = False) -> torch.Tensor:
    # object positions (x,y) plus type id.
    # In-domain: coords ~ N(0,1); OOD: coords shifted/scaled.
    if ood:
        xy = 2.0 * torch.randn(num_obj, 2, generator=g) + torch.tensor([3.0, -2.0])
    else:
        xy = torch.randn(num_obj, 2, generator=g)
    typ = torch.randint(0, 4, (num_obj, 1), generator=g).float()
    scene = torch.cat([xy, typ], dim=1).flatten()  # (num_obj*3)
    return scene


def make_dataset(n: int = 4000, seed: int = 1, ood: bool = False) -> Batch:
    g = torch.Generator().manual_seed(seed)

    scenes = []
    qs = []
    ys = []
    for _ in range(n):
        scene = _make_scene(g, ood=ood)
        # pick two objects by index for relation queries
        i, j = torch.randperm(6, generator=g)[:2].tolist()
        qi = int(torch.randint(0, len(QUESTIONS), (1,), generator=g).item())
        qname = QUESTIONS[qi]

        # decode positions
        s = scene.view(6, 3)
        xi, yi = float(s[i, 0]), float(s[i, 1])
        xj, yj = float(s[j, 0]), float(s[j, 1])

        if qname == "left_of":
            y = 1 if xi < xj else 0
        elif qname == "right_of":
            y = 1 if xi > xj else 0
        elif qname == "closest":
            d = ((s[:, :2] - s[i, :2]) ** 2).sum(dim=1)
            y = int(torch.argmin(d + torch.eye(6)[i] * 1e9).item())  # index of closest other obj
        else:  # count
            y = int((s[:, 2] == s[i, 2]).sum().item()) - 1  # count same-type excluding itself

        # augment img with query object indices (one-hot)
        obj_hint = torch.zeros(12)
        obj_hint[i] = 1.0
        obj_hint[6 + j] = 1.0

        scenes.append(torch.cat([scene, obj_hint], dim=0))
        qs.append(qi)
        ys.append(y)

    img = torch.stack(scenes, dim=0)
    q = torch.tensor(qs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return Batch(img=img, q=q, y=y)


def split(batch: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = batch.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(img=batch.img[i], q=batch.q[i], y=batch.y[i])

    return sel(idx[:k]), sel(idx[k:])
