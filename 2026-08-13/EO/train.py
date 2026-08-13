from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import STEP_INPUT_DIM, create_dataloaders, evaluate_policy
from model import ExperienceOrchestrator, NaiveBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-sessions", type=int, default=1600)
    parser.add_argument("--eval-sessions", type=int, default=256)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, default="eo_toy.pt")
    return parser.parse_args()


@torch.no_grad()
def validate(model: ExperienceOrchestrator, loader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_mae = 0.0
    total_steps = 0
    for batch in loader:
        sequence = batch["sequence"].to(device)
        lengths = batch["length"].to(device)
        action = batch["action"].to(device)
        belief_target = batch["belief_target"].to(device)
        loss, metrics = model.loss(sequence, lengths, action, belief_target)
        total_loss += float(loss.item())
        total_acc += metrics["action_acc"]
        total_mae += metrics["belief_mae"]
        total_steps += 1
    return {
        "val_loss": total_loss / max(1, total_steps),
        "val_action_acc": total_acc / max(1, total_steps),
        "val_belief_mae": total_mae / max(1, total_steps),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_sessions=args.train_sessions,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    model = ExperienceOrchestrator(input_dim=STEP_INPUT_DIM, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = NaiveBaseline()
    baseline_metrics = evaluate_policy(
        baseline,
        num_sessions=args.eval_sessions,
        max_turns=args.max_turns,
        seed=args.seed + 500,
        device=None,
    )

    best_contact = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_mae = 0.0
        steps = 0

        for batch in train_loader:
            sequence = batch["sequence"].to(device)
            lengths = batch["length"].to(device)
            action = batch["action"].to(device)
            belief_target = batch["belief_target"].to(device)

            loss, metrics = model.loss(sequence, lengths, action, belief_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_acc += metrics["action_acc"]
            running_mae += metrics["belief_mae"]
            steps += 1

        val_metrics = validate(model, val_loader, device)
        rollout_metrics = evaluate_policy(
            model,
            num_sessions=args.eval_sessions,
            max_turns=args.max_turns,
            seed=args.seed + 500,
            device=device,
        )
        print(
            f"epoch={epoch} train_loss={running_loss / max(1, steps):.4f} action_acc={running_acc / max(1, steps):.4f} "
            f"belief_mae={running_mae / max(1, steps):.4f} val_loss={val_metrics['val_loss']:.4f} "
            f"val_action_acc={val_metrics['val_action_acc']:.4f} advisor_contact_rate={rollout_metrics['advisor_contact_rate']:.4f} "
            f"genuine_contact_rate={rollout_metrics['genuine_contact_rate']:.4f} avg_resistance_drop={rollout_metrics['avg_resistance_drop']:.4f}"
        )

        if rollout_metrics["genuine_contact_rate"] > best_contact:
            best_contact = rollout_metrics["genuine_contact_rate"]
            best_metrics = rollout_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "input_dim": STEP_INPUT_DIM,
                        "hidden_dim": args.hidden_dim,
                        "max_turns": args.max_turns,
                        "seed": args.seed,
                    },
                    "best_metrics": best_metrics,
                    "baseline_metrics": baseline_metrics,
                },
                checkpoint_path,
            )

    print(f"baseline={baseline_metrics}")
    print(f"best_eo={best_metrics}")
    print(f"saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
