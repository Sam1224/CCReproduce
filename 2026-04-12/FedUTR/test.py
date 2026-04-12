from __future__ import annotations

import argparse
import torch

from data import build_toy_federated_dataset
from model import FedUTRShared, init_clients
from train import _item_token_tensor, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/global.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--users-per-client", type=int, default=50)
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--topics", type=int, default=20)
    parser.add_argument("--sparsity", type=float, default=0.98)
    args = parser.parse_args()

    device = torch.device("cpu")

    dataset = build_toy_federated_dataset(
        seed=args.seed,
        clients=args.clients,
        users_per_client=args.users_per_client,
        items=args.items,
        topics=args.topics,
        sparsity=args.sparsity,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]

    shared = FedUTRShared(
        vocab_size=meta["vocab_size"],
        item_count=meta["item_count"],
        dim=meta["dim"],
        text_dim=meta["dim"],
    ).to(device)
    shared.load_state_dict(ckpt["shared"])

    clients_local = init_clients(
        clients=meta["clients"],
        users_per_client=meta["users_per_client"],
        dim=meta["dim"],
        device=device,
    )
    for cid, state in ckpt["clients"].items():
        cid_int = int(cid)
        clients_local[cid_int].user_emb.load_state_dict(state["user_emb"])
        clients_local[cid_int].user_bias.data.copy_(state["user_bias"].to(device))

    hr, ndcg = evaluate(dataset=dataset, shared=shared, clients_local=clients_local, device=device)
    print(f"HR@10={hr:.4f} NDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    main()
