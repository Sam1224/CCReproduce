from __future__ import annotations

import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import ToyFedDataset, build_toy_federated_dataset
from metrics import evaluate_ranking
from model import FedUTRShared, bpr_loss, init_clients


def _item_token_tensor(item_tokens: List[List[int]], device: torch.device) -> torch.Tensor:
    max_len = max(len(t) for t in item_tokens)
    padded = [t + [0] * (max_len - len(t)) for t in item_tokens]
    return torch.tensor(padded, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate(
    *,
    dataset: ToyFedDataset,
    shared: FedUTRShared,
    clients_local: Dict[int, object],
    k: int = 10,
    device: torch.device,
) -> Tuple[float, float]:
    shared.eval()

    item_tokens = _item_token_tensor(dataset.item_tokens, device)

    hrs: List[float] = []
    ndcgs: List[float] = []
    for client in dataset.clients:
        local = clients_local[client.client_id]
        local.user_emb.eval()

        for uid in range(client.user_count):
            gt = dataset.test_next_item[(client.client_id, uid)]
            user_vec = local.user_emb(torch.tensor([uid], device=device)) + local.user_bias
            user_vec = user_vec.repeat(item_tokens.shape[0], 1)  # [I, D]

            item_ids = torch.arange(item_tokens.shape[0], device=device)
            fused = shared.fused_item_embedding(
                user_vec=user_vec,
                item_id=item_ids,
                item_token_ids=item_tokens,
            )
            scores = (user_vec * fused).sum(dim=-1).detach().cpu().numpy()
            hr, ndcg = evaluate_ranking(scores, gt, k)
            hrs.append(hr)
            ndcgs.append(ndcg)

    return float(np.mean(hrs)), float(np.mean(ndcgs))


def local_train_one_client(
    *,
    dataset: ToyFedDataset,
    shared_global: FedUTRShared,
    client_id: int,
    client_local,
    item_tokens: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
) -> Tuple[Dict[str, torch.Tensor], int]:
    # Clone shared model for local update.
    shared = copy.deepcopy(shared_global).to(device)
    shared.train()
    client_local.user_emb.train()

    # Use a single optimizer over local+shared parameters.
    opt = torch.optim.Adam(
        list(shared.parameters()) + list(client_local.user_emb.parameters()) + [client_local.user_bias],
        lr=lr,
    )

    # Build local interactions list.
    interactions = [pair for pair in dataset.clients[client_id].interactions]
    if not interactions:
        return shared_global.state_dict(), 0

    item_count = len(dataset.item_tokens)
    for _ in range(steps):
        u, pos = random.choice(interactions)
        neg = random.randrange(item_count)
        if neg == pos:
            neg = (neg + 1) % item_count

        u_t = torch.tensor([u], device=device)
        pos_t = torch.tensor([pos], device=device)
        neg_t = torch.tensor([neg], device=device)

        u_vec = client_local.user_emb(u_t) + client_local.user_bias  # [1, D]

        pos_vec = shared.fused_item_embedding(
            user_vec=u_vec,
            item_id=pos_t,
            item_token_ids=item_tokens[pos_t],
        )
        neg_vec = shared.fused_item_embedding(
            user_vec=u_vec,
            item_id=neg_t,
            item_token_ids=item_tokens[neg_t],
        )

        pos_score = (u_vec * pos_vec).sum(dim=-1)
        neg_score = (u_vec * neg_vec).sum(dim=-1)
        loss = bpr_loss(pos_score, neg_score)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return shared.state_dict(), len(interactions)


def fedavg(
    *,
    global_state: Dict[str, torch.Tensor],
    client_states: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    total = sum(w for _, w in client_states)
    if total == 0:
        return global_state

    new_state: Dict[str, torch.Tensor] = {}
    for key in global_state.keys():
        acc = None
        for state, weight in client_states:
            if weight == 0:
                continue
            value = state[key].detach().cpu() * (weight / total)
            acc = value if acc is None else acc + value
        new_state[key] = acc if acc is not None else global_state[key].detach().cpu()
    return new_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--users-per-client", type=int, default=50)
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--topics", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--local-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.98)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    dataset = build_toy_federated_dataset(
        seed=args.seed,
        clients=args.clients,
        users_per_client=args.users_per_client,
        items=args.items,
        topics=args.topics,
        sparsity=args.sparsity,
    )

    item_tokens = _item_token_tensor(dataset.item_tokens, device)

    shared = FedUTRShared(
        vocab_size=dataset.vocab_size,
        item_count=len(dataset.item_tokens),
        dim=args.dim,
        text_dim=args.dim,
    ).to(device)

    clients_local = init_clients(
        clients=args.clients,
        users_per_client=args.users_per_client,
        dim=args.dim,
        device=device,
    )

    # Initial eval
    hr, ndcg = evaluate(dataset=dataset, shared=shared, clients_local=clients_local, device=device)
    print(f"[init] HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    for r in range(1, args.rounds + 1):
        selected = random.sample(range(args.clients), k=min(args.clients_per_round, args.clients))
        client_states: List[Tuple[Dict[str, torch.Tensor], int]] = []

        for cid in selected:
            state, weight = local_train_one_client(
                dataset=dataset,
                shared_global=shared,
                client_id=cid,
                client_local=clients_local[cid],
                item_tokens=item_tokens,
                device=device,
                steps=args.local_steps,
                lr=args.lr,
            )
            client_states.append((state, weight))

        new_state = fedavg(global_state=shared.state_dict(), client_states=client_states)
        shared.load_state_dict({k: v.to(device) for k, v in new_state.items()})

        if r % 5 == 0 or r == args.rounds:
            hr, ndcg = evaluate(dataset=dataset, shared=shared, clients_local=clients_local, device=device)
            print(f"[round {r:02d}] HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "shared": shared.state_dict(),
        "clients": {
            cid: {
                "user_emb": clients_local[cid].user_emb.state_dict(),
                "user_bias": clients_local[cid].user_bias.detach().cpu(),
            }
            for cid in clients_local
        },
        "meta": {
            "vocab_size": dataset.vocab_size,
            "item_count": len(dataset.item_tokens),
            "dim": args.dim,
            "users_per_client": args.users_per_client,
            "clients": args.clients,
        },
    }
    torch.save(ckpt, "checkpoints/global.pt")
    print("saved checkpoints/global.pt")


if __name__ == "__main__":
    main()
