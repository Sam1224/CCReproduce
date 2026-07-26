import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agent import NeuralMemoryAgent, RuleMemoryTracker, StaticInitialAgent, TurnOnlyAgent
from dataset import EvolvingIntentDataset
from metrics import evaluate_agent, format_metrics


def load_neural_agent(checkpoint_path: Path, input_dim: int, hidden_dim: int, num_actions: int, device: torch.device):
    model = NeuralMemoryAgent(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions).to(device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = NeuralMemoryAgent(
            input_dim=checkpoint.get("input_dim", input_dim),
            hidden_dim=checkpoint.get("hidden_dim", hidden_dim),
            num_actions=checkpoint.get("num_actions", num_actions),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded checkpoint from {checkpoint_path}")
    else:
        print(f"checkpoint not found at {checkpoint_path}; evaluating an untrained neural policy")
    return model


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    static_dataset = EvolvingIntentDataset(samples=args.samples, turns=args.turns, drift_prob=0.0, seed=args.seed)
    evolving_dataset = EvolvingIntentDataset(samples=args.samples, turns=args.turns, drift_prob=args.drift_prob, seed=args.seed + 1)
    static_loader = DataLoader(static_dataset, batch_size=args.batch_size)
    evolving_loader = DataLoader(evolving_dataset, batch_size=args.batch_size)

    checkpoint_path = Path(args.checkpoint)
    neural = load_neural_agent(
        checkpoint_path,
        input_dim=evolving_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        num_actions=evolving_dataset.num_actions,
        device=device,
    )
    agents = [StaticInitialAgent(), TurnOnlyAgent(), RuleMemoryTracker(), neural]

    print("Example evolving dialogue:")
    for line in evolving_dataset.render_dialogue(0):
        print("  " + line)

    print("\nStatic no-drift split:")
    for agent in agents:
        print(format_metrics(agent.name, evaluate_agent(agent, static_loader, device)))

    print("\nEvolving-intent split:")
    results = {}
    for agent in agents:
        metrics = evaluate_agent(agent, evolving_loader, device)
        results[agent.name] = metrics
        print(format_metrics(agent.name, metrics))

    static_gap = results["static_initial"]["static_to_evolving_gap"]
    memory_gain = results["rule_memory_tracker"]["evolving_acc"] - results["static_initial"]["evolving_acc"]
    print(
        "\nStory check: static_initial is perfect at turn-0/static reading, "
        f"but loses {static_gap:.3f} accuracy under evolving intent; "
        f"explicit memory recovers {memory_gain:.3f} evolving accuracy."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--drift-prob", type=float, default=0.8)
    parser.add_argument("--checkpoint", default="outputs/neural_memory.pt")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
