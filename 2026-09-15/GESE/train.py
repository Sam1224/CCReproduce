from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    PairDataset,
    SelectorDataset,
    build_splits,
    build_world,
    diversity_reward,
    reward_batch,
    set_seed,
)
from model import Generator, Selector


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_generator(
    world,
    train_loader: DataLoader,
    n_users: int,
    n_topics: int,
    num_candidates: int = 8,
    epochs: int = 15,
    lr: float = 3e-3,
    lambda_div: float = 0.25,
) -> tuple[Generator, list[dict]]:
    gen = Generator(n_users=n_users, n_topics=n_topics, hidden_dim=24, emb_dim=24).to(DEVICE)
    opt = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        gen.train()
        loss_sum = 0.0
        ctr_sum = 0.0
        div_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            user_id = batch["user_id"].to(DEVICE)
            item_id = batch["item_id"].to(DEVICE)
            item_topic = world.item_topic[item_id]

            # 1) exploration: sample K candidates
            with torch.no_grad():
                sample_out = gen.sample(user_id, item_topic, num_candidates=num_candidates, temperature=1.25)
                titles = sample_out.titles  # [B,K,L]

            bsz, k, l = titles.shape
            flat_titles = titles.view(bsz * k, l)
            flat_user = user_id.repeat_interleave(k)
            flat_item = item_id.repeat_interleave(k)
            flat_topic = item_topic.repeat_interleave(k)

            # 2) env reward (CTR proxy)
            base_ctr = reward_batch(world, flat_user, flat_item, flat_titles).view(bsz, k)

            # 3) diversity reward (set-level) and combine
            div = diversity_reward(titles.cpu()).to(DEVICE)  # [B]
            total_reward = base_ctr + lambda_div * div.unsqueeze(1)
            baseline = total_reward.mean(dim=1, keepdim=True)
            advantage = (total_reward - baseline).detach()

            # 4) reinforce objective uses differentiable log-prob
            logp = gen.log_prob(flat_user, flat_topic, flat_titles).view(bsz, k)
            loss = -(advantage * logp).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            opt.step()

            loss_sum += loss.item()
            ctr_sum += base_ctr.mean().item()
            div_sum += div.mean().item()
            n_batches += 1

        history.append(
            {
                "epoch": epoch,
                "loss": round(loss_sum / max(1, n_batches), 4),
                "avg_ctr": round(ctr_sum / max(1, n_batches), 4),
                "avg_div": round(div_sum / max(1, n_batches), 4),
            }
        )

    return gen, history


@torch.no_grad()
def build_selector_dataset(
    world,
    gen: Generator,
    loader: DataLoader,
    num_candidates: int = 8,
) -> SelectorDataset:
    gen.eval()

    all_user = []
    all_item = []
    all_title = []
    all_reward = []

    for batch in loader:
        user_id = batch["user_id"].to(DEVICE)
        item_id = batch["item_id"].to(DEVICE)
        item_topic = world.item_topic[item_id]

        sample_out = gen.sample(user_id, item_topic, num_candidates=num_candidates, temperature=1.1)
        titles = sample_out.titles  # [B,K,L]

        bsz, k, l = titles.shape
        flat_titles = titles.view(bsz * k, l)
        flat_user = user_id.repeat_interleave(k)
        flat_item = item_id.repeat_interleave(k)

        rewards = reward_batch(world, flat_user, flat_item, flat_titles)

        all_user.append(flat_user.cpu())
        all_item.append(flat_item.cpu())
        all_title.append(flat_titles.cpu())
        all_reward.append(rewards.cpu())

    return SelectorDataset(
        user_ids=torch.cat(all_user, dim=0),
        item_ids=torch.cat(all_item, dim=0),
        titles=torch.cat(all_title, dim=0),
        rewards=torch.cat(all_reward, dim=0),
    )


def train_selector(
    world,
    dataset: SelectorDataset,
    n_users: int,
    n_topics: int,
    epochs: int = 8,
    lr: float = 2e-3,
) -> tuple[Selector, list[dict]]:
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    selector = Selector(n_users=n_users, n_topics=n_topics, hidden_dim=48, emb_dim=24).to(DEVICE)
    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        selector.train()
        loss_sum = 0.0
        for batch in loader:
            user_id = batch["user_id"].to(DEVICE)
            item_id = batch["item_id"].to(DEVICE)
            item_topic = world.item_topic[item_id]
            title = batch["title"].to(DEVICE)
            target = batch["reward"].to(DEVICE)

            pred = selector(user_id, item_topic, title)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += loss.item()

        history.append({"epoch": epoch, "mse": round(loss_sum / max(1, len(loader)), 6)})

    return selector, history


def main() -> None:
    set_seed(7)

    # Synthetic environment config
    cfg = {
        "seed": 7,
        "n_users": 64,
        "n_items": 200,
        "n_train": 800,
        "n_test": 200,
        "gen_candidates": 6,
        "sel_candidates": 6,
        "gen_epochs": 6,
        "sel_epochs": 4,
    }

    world = build_world(n_users=cfg["n_users"], n_items=cfg["n_items"], seed=cfg["seed"])  # CPU
    # Move world tensors to DEVICE for reward_batch
    world.user_topic_pref = world.user_topic_pref.to(DEVICE)
    world.user_style_pref = world.user_style_pref.to(DEVICE)
    world.item_topic = world.item_topic.to(DEVICE)

    splits = build_splits(
        seed=cfg["seed"],
        n_users=cfg["n_users"],
        n_items=cfg["n_items"],
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
    )

    train_loader = DataLoader(PairDataset(splits.train_pairs), batch_size=64, shuffle=True, drop_last=True)
    train_loader_noshuf = DataLoader(PairDataset(splits.train_pairs), batch_size=128)

    gen, gen_hist = train_generator(
        world,
        train_loader=train_loader,
        n_users=cfg["n_users"],
        n_topics=world.n_topics,
        num_candidates=cfg["gen_candidates"],
        epochs=cfg["gen_epochs"],
    )

    torch.save({"state_dict": gen.state_dict(), "cfg": cfg}, ROOT / "generator.pt")

    selector_ds = build_selector_dataset(
        world,
        gen=gen,
        loader=train_loader_noshuf,
        num_candidates=cfg["sel_candidates"],
    )

    selector, sel_hist = train_selector(
        world,
        selector_ds,
        n_users=cfg["n_users"],
        n_topics=world.n_topics,
        epochs=cfg["sel_epochs"],
    )
    torch.save({"state_dict": selector.state_dict(), "cfg": cfg}, ROOT / "selector.pt")

    out = {
        "device": str(DEVICE),
        "cfg": cfg,
        "generator_train": gen_hist,
        "selector_train": sel_hist,
        "selector_dataset_size": len(selector_ds),
    }
    (ROOT / "train_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "device": str(DEVICE), "selector_dataset": len(selector_ds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
