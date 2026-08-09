import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ToyAgentConfig, ToyAgentTrajectoryDataset, collate_trajectories
from model import AgentOPSDConfig, AgentOPSDCredit, TinyAgentPolicy, weighted_policy_loss


def train(args):
    torch.manual_seed(args.seed)
    data_config = ToyAgentConfig(num_samples=args.samples, turns=args.turns, seed=args.seed)
    dataset = ToyAgentTrajectoryDataset(data_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trajectories)

    model_config = AgentOPSDConfig(vocab_size=data_config.vocab_size, num_actions=data_config.num_actions)
    policy = TinyAgentPolicy(model_config)
    credit_model = AgentOPSDCredit(model_config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        for batch in loader:
            logits = policy(batch["observations"])
            action_log_probs = F.log_softmax(logits, dim=-1).gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
            turn_credit, _ = credit_model(
                batch["teacher_log_probs"],
                batch["student_log_probs"],
                batch["token_mask"],
                batch["reward"],
            )
            opsd_loss = weighted_policy_loss(action_log_probs, batch["reward"], turn_credit)
            imitation_loss = F.cross_entropy(logits.reshape(-1, model_config.num_actions), batch["target_actions"].reshape(-1))
            loss = imitation_loss + args.opsd_weight * opsd_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                accuracy = (logits.argmax(dim=-1) == batch["target_actions"]).float().mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()

        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f} action_acc={total_accuracy / len(loader):.4f}")

    return policy


def parse_args():
    parser = argparse.ArgumentParser(description="Toy AgentOPSD training pipeline")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--opsd-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
