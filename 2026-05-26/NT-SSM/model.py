from __future__ import annotations

import torch
from torch import nn


class LightGCN(nn.Module):
    """Toy LightGCN using a precomputed \tilde{S} matrix.

    For the toy setting, we compute final embeddings as:
      E = \tilde{S} E0
    where E0 are learnable node embeddings.
    """

    def __init__(self, num_nodes: int, dim: int, s_tilde: torch.Tensor):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim

        self.emb0 = nn.Parameter(torch.randn(num_nodes, dim) * 0.01)
        self.register_buffer("s_tilde", s_tilde)

        # Column-masked versions for neighbor-type decomposition
        # user nodes are [0, U), item nodes are [U, U+I)
        self.register_buffer("mask_user", torch.zeros(num_nodes))
        self.register_buffer("mask_item", torch.zeros(num_nodes))

    def set_type_masks(self, *, num_users: int) -> None:
        self.mask_user.zero_()
        self.mask_item.zero_()
        self.mask_user[:num_users] = 1.0
        self.mask_item[num_users:] = 1.0

    def forward_all(self) -> dict[str, torch.Tensor]:
        # E_all = S @ E0
        e_all = self.s_tilde @ self.emb0

        # E_from_users = (S * mask_user_cols) @ E0
        s_user = self.s_tilde * self.mask_user.unsqueeze(0)
        s_item = self.s_tilde * self.mask_item.unsqueeze(0)
        e_from_users = s_user @ self.emb0
        e_from_items = s_item @ self.emb0

        return {"e_all": e_all, "e_from_users": e_from_users, "e_from_items": e_from_items}
