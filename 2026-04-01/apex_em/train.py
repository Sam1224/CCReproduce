from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToyAgentEpisodeDataset
from model import APEXEMToyPolicy, EpisodicMemory


def build_memory(ds: ToyAgentEpisodeDataset) -> EpisodicMemory:
    states = []
    actions = []
    for i in range(len(ds)):
        x = ds[i]
        states.append(x.state)
        actions.append(x.action)
    mem = EpisodicMemory(state_dim=ds.state_dim)
    mem.build(states=torch.stack(states, dim=0), actions=torch.stack(actions, dim=0))
    return mem


def memory_action_dist(mem: EpisodicMemory, *, state: torch.Tensor, num_actions: int, top_k: int = 8) -> torch.Tensor:
    res = mem.query(state, top_k=top_k)
    b = state.shape[0]
    dist = torch.zeros((b, num_actions), dtype=torch.float32)
    for i in range(res.neighbor_actions.shape[1]):
        a = res.neighbor_actions[:, i]
        w = res.neighbor_weights[:, i]
        dist.scatter_add_(1, a.view(-1, 1), w.view(-1, 1))
    dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="ckpt.pt")
    args = ap.parse_args()

    device = torch.device("cpu")

    train_ds = ToyAgentEpisodeDataset(seed=0)
    val_ds = ToyAgentEpisodeDataset(seed=1)

    mem = build_memory(train_ds)

    model = APEXEMToyPolicy(state_dim=train_ds.state_dim, num_actions=train_ds.num_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            mem_dist = memory_action_dist(mem, state=batch.state, num_actions=train_ds.num_actions)
            logp = model(state=batch.state.to(device), memory_dist=mem_dist.to(device))
            loss = F.nll_loss(logp, batch.action.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                mem_dist = memory_action_dist(mem, state=batch.state, num_actions=train_ds.num_actions)
                logp = model(state=batch.state.to(device), memory_dist=mem_dist.to(device))
                pred = logp.argmax(dim=-1)
                correct += (pred.cpu() == batch.action).sum().item()
                total += batch.action.numel()
        acc = correct / max(1, total)
        print(f"epoch={epoch} val_acc={acc:.4f}")

    out = Path(args.out)
    torch.save({"model": model.state_dict()}, out)


if __name__ == "__main__":
    main()
