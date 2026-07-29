import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyLiveStreamDataset, collate_streams
from model import ContentPolicy, FrozenTwoTowerScorer, group_relative_policy_loss, recommender_affinity_reward


def train(args):
    torch.manual_seed(args.seed)
    dataset = ToyLiveStreamDataset(num_items=args.num_items, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_streams)
    policy = ContentPolicy()
    scorer = FrozenTwoTowerScorer()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_reward = 0.0
        for batch in loader:
            sampled = policy.sample_descriptions(batch["content"], rollouts=args.rollouts)
            reward = recommender_affinity_reward(
                sampled["description"],
                batch["target_users"],
                batch["non_target_users"],
                scorer,
                lambda_non_target=args.lambda_non_target,
            )
            loss = group_relative_policy_loss(sampled["logprob"], reward)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_reward += reward.mean().item()
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f} reward={total_reward / len(loader):.4f}")
    torch.save(policy.state_dict(), args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--lambda-non-target", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="recoreward_policy.pt")
    train(parser.parse_args())
