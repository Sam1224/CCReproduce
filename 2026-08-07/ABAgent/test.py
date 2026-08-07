from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean
from typing import List

import torch

from dataset import MECHANISMS, RequestCase
from evolution import ExperimentNode, ExperimentTree, StrategyConfig, compute_utility
from experience_tree import ExperienceTree
from model import StrategyGenerator, StrategyValueNet, ValueNetConfig
from retriever import DenseRetriever, HybridRetriever, Reranker
from simulator import ABSimulator


def load_components(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = payload["vocab"]
    vectorizer = payload["vectorizer"]
    chunks = payload["chunks"]
    requests = payload["requests"]

    dense_model = DenseRetriever(len(vocab)).to(device)
    dense_model.load_state_dict(payload["dense_model"])
    dense_model.eval()

    reranker = Reranker().to(device)
    reranker.load_state_dict(payload["reranker"])
    reranker.eval()

    value_net = StrategyValueNet(ValueNetConfig()).to(device)
    value_net.load_state_dict(payload["value_net"])
    value_net.eval()

    tree = ExperienceTree(chunks)
    chunk_matrix = vectorizer.transform([chunk.text for chunk in chunks])
    retriever = HybridRetriever(chunks, tree, vocab, dense_model, reranker, vectorizer, chunk_matrix, device)
    return requests, retriever, value_net


def random_baseline(request: RequestCase, simulator: ABSimulator) -> float:
    best = -999.0
    rng = random.Random(hash(request.request_id) % 10000)
    for trial in range(6):
        strategy = StrategyConfig(mechanism=rng.choice(MECHANISMS), params=[rng.random() for _ in range(4)])
        observed = simulator.run(request, strategy, step=trial)
        best = max(best, compute_utility(request, observed))
    return best


def run_agent(request: RequestCase, retriever: HybridRetriever, generator: StrategyGenerator, simulator: ABSimulator, steps: int) -> float:
    ranked = retriever.retrieve(request, topk=8)
    current = generator.refine(request, generator.propose(request, ranked))
    observed = simulator.run(request, current, step=0)
    utility = compute_utility(request, observed)
    history = ExperimentTree(ExperimentNode(step=0, strategy=current, observed_metrics=observed, utility=utility, parent_index=None))
    for step in range(1, steps + 1):
        best = history.best()
        proposals: List[StrategyConfig] = [generator.refine(request, best.strategy, steps=8, lr=0.03)]
        for candidate in ranked[:3]:
            candidate_strategy = StrategyConfig(mechanism=candidate["chunk"].mechanism, params=list(candidate["chunk"].params))
            proposals.append(generator.refine(request, candidate_strategy, steps=8, lr=0.03))
        if best.observed_metrics["creator_fairness"] < 0.0:
            proposals.append(StrategyConfig(best.strategy.mechanism, [best.strategy.params[0], best.strategy.params[1], min(1.0, best.strategy.params[2] + 0.10), min(1.0, best.strategy.params[3] + 0.05)]))
        if best.observed_metrics["refund_safety"] < 0.0:
            proposals.append(StrategyConfig(best.strategy.mechanism, [max(0.0, best.strategy.params[0] - 0.10), best.strategy.params[1], best.strategy.params[2], min(1.0, best.strategy.params[3] + 0.15)]))
        scored = []
        for proposal in proposals:
            observed = simulator.run(request, proposal, step=step)
            utility = compute_utility(request, observed)
            scored.append((utility, proposal, observed))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_utility, best_strategy, best_observed = scored[0]
        history.add(ExperimentNode(step=step, strategy=best_strategy, observed_metrics=best_observed, utility=best_utility, parent_index=0))
    return history.best().utility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/ab_agent.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    requests, retriever, value_net = load_components(checkpoint_path, device)
    simulator = ABSimulator(seed=13)
    generator = StrategyGenerator(value_net, device)

    random_scores = []
    init_scores = []
    evolve_scores = []

    for request in requests:
        ranked = retriever.retrieve(request, topk=8)
        init_strategy = generator.refine(request, generator.propose(request, ranked))
        init_observed = simulator.run(request, init_strategy, step=0)
        init_utility = compute_utility(request, init_observed)
        evolve_utility = run_agent(request, retriever, generator, simulator, steps=4)
        random_utility = random_baseline(request, simulator)
        random_scores.append(random_utility)
        init_scores.append(init_utility)
        evolve_scores.append(evolve_utility)
        print(f"{request.request_id} | random={random_utility:.4f} | init={init_utility:.4f} | evolve={evolve_utility:.4f}")

    print(f"Average random utility: {mean(random_scores):.4f}")
    print(f"Average init utility:   {mean(init_scores):.4f}")
    print(f"Average evolve utility: {mean(evolve_scores):.4f}")


if __name__ == "__main__":
    main()
