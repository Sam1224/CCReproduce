from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CatalogItem, UserExample, default_synthetic_setup
from model import QueryAgentR1Toy, Vocab, sample_actions, supervised_loss


def build_item_token_tensor(vocab: Vocab, catalog: list[CatalogItem]) -> torch.Tensor:
    ids = [vocab.encode(it.tokens) for it in catalog]
    return torch.tensor(ids, dtype=torch.long)


def build_history_token_tensor(item_token_ids: torch.Tensor, histories: list[list[int]]) -> torch.Tensor:
    # histories: list of [H] item_ids -> [B, H, T]
    tokens = [item_token_ids[torch.tensor(h, dtype=torch.long)] for h in histories]
    return torch.stack(tokens, dim=0)


def build_target_query_tensor(vocab: Vocab, targets: list[list[str]], max_len: int) -> torch.Tensor:
    tok = [vocab.encode(t)[:max_len] for t in targets]
    tok = [t + [vocab.token_to_id["<pad>"]] * (max_len - len(t)) for t in tok]
    return torch.tensor(tok, dtype=torch.long)


def compute_reward(
    *,
    actions: torch.Tensor,
    targets: torch.Tensor,
    top1: torch.Tensor,
    purchase_sets: list[set[int]],
    vocab: Vocab,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Q_EM: exact match of tokens (pad ignored)
    pad = vocab.token_to_id["<pad>"]
    mask = targets != pad
    q_em = ((actions == targets) | ~mask).all(dim=1).float()

    i_hit = torch.tensor(
        [1.0 if int(top1[i].item()) in purchase_sets[i] else 0.0 for i in range(len(purchase_sets))],
        dtype=torch.float,
        device=actions.device,
    )
    cons = q_em * i_hit

    # Consistency reward (toy): emphasize end-to-end alignment
    reward = 0.5 * q_em + 0.5 * i_hit + 1.0 * cons

    stats = {
        "Q_EM": float(q_em.mean().item()),
        "I_Hit@1": float(i_hit.mean().item()),
        "Cons@1": float(cons.mean().item()),
        "Reward": float(reward.mean().item()),
    }
    return reward, stats


def save_checkpoint(path: str, model: QueryAgentR1Toy, vocab: Vocab) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": asdict(vocab),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sup-epochs", type=int, default=3)
    parser.add_argument("--rl-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs/queryagent_r1.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    catalog, train, val, _ = default_synthetic_setup(seed=args.seed)

    vocab = Vocab.from_catalog([it.tokens for it in catalog])
    item_token_ids = build_item_token_tensor(vocab, catalog)

    device = torch.device(args.device)

    model = QueryAgentR1Toy(
        n_items=len(catalog),
        vocab_size=len(vocab.id_to_token),
        embed_dim=args.embed_dim,
        item_token_ids=item_token_ids,
        max_query_len=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def make_loader(examples: list[UserExample], shuffle: bool) -> DataLoader:
        # Keep batches as raw UserExample objects (toy dataset), avoid default tensor collation.
        return DataLoader(examples, batch_size=args.batch_size, shuffle=shuffle, drop_last=False, collate_fn=lambda x: x)

    # ---------------------------
    # Stage 0: supervised warmup
    # ---------------------------
    for epoch in range(args.sup_epochs):
        model.train()
        losses = []
        for batch in make_loader(train, shuffle=True):
            histories = [ex.history for ex in batch]
            targets = [ex.target_query_tokens for ex in batch]

            hist_tok = build_history_token_tensor(item_token_ids, histories).to(device)
            tgt_tok = build_target_query_tensor(vocab, targets, max_len=2).to(device)

            user_state = model.user_state(hist_tok)
            logits, _ = model.policy(user_state)
            loss = supervised_loss(logits, tgt_tok)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        print(f"[sup] epoch={epoch+1}/{args.sup_epochs} loss={sum(losses)/len(losses):.4f}")

    # ---------------------------
    # Stage 1: RL fine-tuning
    # ---------------------------
    for epoch in range(args.rl_epochs):
        model.train()
        stats_accum = {"Q_EM": [], "I_Hit@1": [], "Cons@1": [], "Reward": [], "Loss": []}
        for batch in make_loader(train, shuffle=True):
            histories = [ex.history for ex in batch]
            targets = [ex.target_query_tokens for ex in batch]
            purchase_sets = [ex.purchase_set for ex in batch]

            hist_tok = build_history_token_tensor(item_token_ids, histories).to(device)
            tgt_tok = build_target_query_tensor(vocab, targets, max_len=2).to(device)

            user_state = model.user_state(hist_tok)
            logits, value = model.policy(user_state)
            actions, logp = sample_actions(logits)

            # Retrieval grounding: compute top-1 retrieval and reward
            top_items = model.retrieve(actions, item_token_ids, topk=args.topk)
            top1 = top_items[:, 0]

            reward, stats = compute_reward(actions=actions, targets=tgt_tok, top1=top1, purchase_sets=purchase_sets, vocab=vocab)

            advantage = reward.detach() - value.detach()
            policy_loss = -(logp * advantage).mean()
            value_loss = torch.mean((value - reward.detach()) ** 2)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            for k in ["Q_EM", "I_Hit@1", "Cons@1", "Reward"]:
                stats_accum[k].append(stats[k])
            stats_accum["Loss"].append(float(loss.item()))

        print(
            "[rl] "
            + " ".join(
                [
                    f"epoch={epoch+1}/{args.rl_epochs}",
                    f"Q_EM={np.mean(stats_accum['Q_EM']):.3f}",
                    f"I_Hit@1={np.mean(stats_accum['I_Hit@1']):.3f}",
                    f"Cons@1={np.mean(stats_accum['Cons@1']):.3f}",
                    f"Reward={np.mean(stats_accum['Reward']):.3f}",
                    f"loss={np.mean(stats_accum['Loss']):.3f}",
                ]
            )
        )

    save_checkpoint(args.out, model, vocab)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
