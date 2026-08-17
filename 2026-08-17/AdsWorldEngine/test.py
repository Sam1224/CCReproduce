from __future__ import annotations

from pathlib import Path

import torch

from data import benchmark_conversations, build_ads, dialog_to_bow
from model import AdsWorldEnginePipeline, OpportunityGateNet, RelevanceJudge, SlateOrchestrator


def balanced_metrics(labels, preds):
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "balanced_accuracy": round((tpr + tnr) / 2.0, 4),
    }


def main() -> None:
    ads = build_ads()
    conversations = benchmark_conversations()
    checkpoint = torch.load(Path(__file__).resolve().parent / "adsworldengine_toy.pt", map_location="cpu")

    gate = OpportunityGateNet(checkpoint["dialog_dim"])
    gate.load_state_dict(checkpoint["gate"])
    judge = RelevanceJudge(checkpoint["dialog_dim"], checkpoint["ad_dim"], checkpoint["tool_dim"])
    judge.load_state_dict(checkpoint["judge"])
    orchestrator = SlateOrchestrator(checkpoint["dialog_dim"], checkpoint["ad_dim"], checkpoint["tool_dim"])
    orchestrator.load_state_dict(checkpoint["orchestrator"])

    pipeline = AdsWorldEnginePipeline(gate.eval(), judge.eval(), orchestrator.eval(), ads)

    labels = []
    preds = []
    rewards = []
    relevances = []
    diversity = []

    for example in conversations:
        gate_prob, slate = pipeline.predict(example, top_k=3)
        pred = 1 if gate_prob >= 0.5 else 0
        labels.append(example.trigger_label)
        preds.append(pred)
        if slate:
            rewards.append(pipeline.reward(example, slate))
            relevances.append(pipeline.slate_relevance(example, slate))
            diversity.append(len({ad.brand for ad in slate}) / len(slate))

    metrics = balanced_metrics(labels, preds)
    metrics.update(
        {
            "avg_reward": round(sum(rewards) / max(len(rewards), 1), 4),
            "avg_relevance": round(sum(relevances) / max(len(relevances), 1), 4),
            "avg_brand_diversity": round(sum(diversity) / max(len(diversity), 1), 4),
            "evaluated_cases": len(conversations),
        }
    )
    print(metrics)


if __name__ == "__main__":
    main()
