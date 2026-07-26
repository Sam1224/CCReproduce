import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agent import NeuralMemoryAgent, RuleMemoryTracker, StaticInitialAgent, TurnOnlyAgent
from dataset import EvolvingIntentDataset
from metrics import evaluate_agent, format_metrics


def train_one_epoch(model: NeuralMemoryAgent, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        utterances = batch["utterances"].to(device)
        labels = batch["action_labels"].to(device)
        logits = model(utterances)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_dataset = EvolvingIntentDataset(
        samples=args.samples,
        turns=args.turns,
        drift_prob=args.train_drift_prob,
        seed=args.seed,
    )
    dev_evolving = EvolvingIntentDataset(
        samples=max(128, args.samples // 3),
        turns=args.turns,
        drift_prob=args.eval_drift_prob,
        seed=args.seed + 1,
    )
    dev_static = EvolvingIntentDataset(
        samples=max(128, args.samples // 3),
        turns=args.turns,
        drift_prob=0.0,
        seed=args.seed + 2,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    evolving_loader = DataLoader(dev_evolving, batch_size=args.batch_size)
    static_loader = DataLoader(dev_static, batch_size=args.batch_size)

    model = NeuralMemoryAgent(
        input_dim=train_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        num_actions=train_dataset.num_actions,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    baselines = [StaticInitialAgent(), TurnOnlyAgent(), RuleMemoryTracker()]
    print("Before training on evolving split:")
    for agent in baselines + [model]:
        print(format_metrics(agent.name, evaluate_agent(agent, evolving_loader, device)))

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_agent(model, evolving_loader, device)
        print(
            f"epoch={epoch + 1} loss={loss:.4f} "
            f"evolving_acc={metrics['evolving_acc']:.3f} final_acc={metrics['final_acc']:.3f}"
        )

    print("\nStatic no-drift evaluation:")
    for agent in baselines + [model]:
        print(format_metrics(agent.name, evaluate_agent(agent, static_loader, device)))

    print("\nEvolving-intent evaluation:")
    for agent in baselines + [model]:
        print(format_metrics(agent.name, evaluate_agent(agent, evolving_loader, device)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "neural_memory.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": train_dataset.feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_actions": train_dataset.num_actions,
            "turns": args.turns,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-drift-prob", type=float, default=0.7)
    parser.add_argument("--eval-drift-prob", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
