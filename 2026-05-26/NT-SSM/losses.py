from __future__ import annotations

import torch


def ssm_loss(
    *,
    pos_score: torch.Tensor,  # (B,)
    neg_score: torch.Tensor,  # (B, Nneg)
    tau: float,
) -> torch.Tensor:
    # -log exp(pos) / (exp(pos) + sum exp(neg))
    pos = pos_score / tau
    neg = neg_score / tau
    denom = torch.logsumexp(torch.cat([pos.unsqueeze(1), neg], dim=1), dim=1)
    return -(pos - denom).mean()


def nt_ssm_loss(
    *,
    pos_score: torch.Tensor,
    neg_item_score: torch.Tensor,
    neg_user_score: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    # L(u,i) = L(i;u) + L(u;i)
    l_item = ssm_loss(pos_score=pos_score, neg_score=neg_item_score, tau=tau)
    l_user = ssm_loss(pos_score=pos_score, neg_score=neg_user_score, tau=tau)
    return l_item + l_user


def compute_scores_for_batch(
    *,
    emb: dict[str, torch.Tensor],
    user_nodes: torch.Tensor,  # (B,)
    item_nodes: torch.Tensor,  # (B,)
    neg_item_nodes: torch.Tensor,  # (B, Nneg)
    neg_user_nodes: torch.Tensor,  # (B, Nneg)
    alpha_I_U: float,
    alpha_I_I: float,
    alpha_U_U: float,
    alpha_U_I: float,
) -> dict[str, torch.Tensor]:
    """Compute positive score and negative scores for SSM / NT-SSM.

    Notation aligns with the paper:
    - positive score uses s(u,i) = e_u^T e_i
    - NT-SSM uses type-decomposed negative similarity:
      \tilde{s}(u,j) = alpha_I^U * (e_u^T e_{j,from_users}) + alpha_I^I * (e_u^T e_{j,from_items})
      \tilde{s}(i,k) = alpha_U^U * (e_i^T e_{k,from_users}) + alpha_U^I * (e_i^T e_{k,from_items})
    """

    e_all = emb["e_all"]
    e_from_users = emb["e_from_users"]
    e_from_items = emb["e_from_items"]

    e_u = e_all[user_nodes]  # (B, D)
    e_i = e_all[item_nodes]  # (B, D)

    pos = (e_u * e_i).sum(dim=1)

    # SSM negative items/users use the same final embedding dot
    e_neg_items = e_all[neg_item_nodes]  # (B, N, D)
    e_neg_users = e_all[neg_user_nodes]

    ssm_neg_items = (e_u.unsqueeze(1) * e_neg_items).sum(dim=2)
    ssm_neg_users = (e_i.unsqueeze(1) * e_neg_users).sum(dim=2)

    # NT-SSM negative similarity decomposition
    e_neg_items_u = e_from_users[neg_item_nodes]
    e_neg_items_i = e_from_items[neg_item_nodes]
    nt_neg_items = alpha_I_U * (e_u.unsqueeze(1) * e_neg_items_u).sum(dim=2) + alpha_I_I * (e_u.unsqueeze(1) * e_neg_items_i).sum(dim=2)

    e_neg_users_u = e_from_users[neg_user_nodes]
    e_neg_users_i = e_from_items[neg_user_nodes]
    nt_neg_users = alpha_U_U * (e_i.unsqueeze(1) * e_neg_users_u).sum(dim=2) + alpha_U_I * (e_i.unsqueeze(1) * e_neg_users_i).sum(dim=2)

    return {
        "pos": pos,
        "ssm_neg_items": ssm_neg_items,
        "ssm_neg_users": ssm_neg_users,
        "nt_neg_items": nt_neg_items,
        "nt_neg_users": nt_neg_users,
    }
