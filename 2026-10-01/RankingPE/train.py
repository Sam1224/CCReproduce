from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import GovernanceDataset, build_splits, positive_negative_pairs, set_seed
from model import ToyMultimodalJudge, build_prompt_candidates, score_with_prompts

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pairs = positive_negative_pairs(labels.cpu(), max_pairs=20000, seed=99)
    if not pairs:
        return 0.5
    correct = 0.0
    for pos_idx, neg_idx in pairs:
        correct += float(scores[pos_idx] > scores[neg_idx])
        correct += 0.5 * float(scores[pos_idx] == scores[neg_idx])
    return correct / len(pairs)


def balanced_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (scores.sigmoid() >= 0.5).long()
    pos = labels == 1
    neg = labels == 0
    tpr = (preds[pos] == 1).float().mean().item() if pos.any() else 0.0
    tnr = (preds[neg] == 0).float().mean().item() if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def train_base_model(model: ToyMultimodalJudge, train_loader: DataLoader, epochs: int = 8) -> List[Dict[str, float]]:
    prompts = build_prompt_candidates(DEVICE)
    neutral = prompts[-1].feature_prior.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            features = batch["features"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE)
            logits = model(features, neutral)
            pos_weight = torch.tensor([5.0], device=DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
        history.append({"epoch": epoch, "loss": round(loss_sum / max(1, len(train_loader)), 5)})
    return history


@torch.no_grad()
def select_prompts(model: ToyMultimodalJudge, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, Dict[str, object]]:
    prompts = build_prompt_candidates(features.device)
    matrix = score_with_prompts(model, features, prompts)
    accuracy_scores = []
    ranking_scores = []
    pairs = positive_negative_pairs(labels.cpu(), max_pairs=6000, seed=17)
    for col in range(matrix.shape[1]):
        logits = matrix[:, col]
        pred = (logits.sigmoid() >= 0.5).long()
        accuracy_scores.append((pred == labels).float().mean().item())
        pair_hits = [float(logits[p] > logits[n]) + 0.5 * float(logits[p] == logits[n]) for p, n in pairs]
        ranking_scores.append(sum(pair_hits) / max(1, len(pair_hits)))

    acc_idx = int(torch.tensor(accuracy_scores).argmax().item())
    rank_idx = int(torch.tensor(ranking_scores).argmax().item())
    return {
        "accuracy_pe": {
            "prompt": prompts[acc_idx].name,
            "instruction_zh": prompts[acc_idx].instruction_zh,
            "selection_score": round(accuracy_scores[acc_idx], 4),
            "all_scores": [round(v, 4) for v in accuracy_scores],
        },
        "ranking_pe": {
            "prompt": prompts[rank_idx].name,
            "instruction_zh": prompts[rank_idx].instruction_zh,
            "selection_score": round(ranking_scores[rank_idx], 4),
            "all_scores": [round(v, 4) for v in ranking_scores],
        },
    }


@torch.no_grad()
def evaluate_prompt(model: ToyMultimodalJudge, features: torch.Tensor, labels: torch.Tensor, prompt_name: str) -> Dict[str, float]:
    prompts = build_prompt_candidates(features.device)
    selected = next(p for p in prompts if p.name == prompt_name)
    logits = model(features, selected.feature_prior.to(features.device))
    return {
        "auroc": round(auroc(logits.cpu(), labels.cpu()), 4),
        "balanced_accuracy": round(balanced_accuracy(logits.cpu(), labels.cpu()), 4),
        "positive_rate_pred": round(float((logits.sigmoid() >= 0.5).float().mean().item()), 4),
    }


def main() -> None:
    set_seed(13)
    splits = build_splits(seed=13)
    train_loader = DataLoader(GovernanceDataset(splits.train), batch_size=64, shuffle=True)
    model = ToyMultimodalJudge(hidden_dim=48).to(DEVICE)
    history = train_base_model(model, train_loader, epochs=8)

    valid_features = splits.valid.features.to(DEVICE)
    valid_labels = splits.valid.labels.to(DEVICE)
    selection = select_prompts(model, valid_features, valid_labels)

    test_features = splits.test.features.to(DEVICE)
    test_labels = splits.test.labels.to(DEVICE)
    results = {
        "paper": "Ranking-Aware Prompt Optimization for Multimodal Clinical Diagnosis",
        "toy_domain": "e-commerce content and creator-governance risk ranking",
        "training_history": history,
        "selection": selection,
        "test": {
            "accuracy_pe": evaluate_prompt(model, test_features, test_labels, selection["accuracy_pe"]["prompt"]),
            "ranking_pe": evaluate_prompt(model, test_features, test_labels, selection["ranking_pe"]["prompt"]),
        },
    }
    torch.save({"state_dict": model.state_dict()}, ROOT / "ranking_pe_model.pt")
    (ROOT / "train_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
