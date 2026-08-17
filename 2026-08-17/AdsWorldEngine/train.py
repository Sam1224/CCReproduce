from __future__ import annotations

from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import (
    GateDataset,
    JudgeDataset,
    OrchestratorDataset,
    ad_to_vector,
    build_ads,
    build_conversations,
    collate_gate,
    collate_judge,
    dialog_to_bow,
    slate_reward,
    tool_features,
)
from model import OpportunityGateNet, RelevanceJudge, SlateOrchestrator


def train_gate(gate: OpportunityGateNet, dataset: GateDataset) -> None:
    train_size = int(len(dataset) * 0.85)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_gate)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, collate_fn=collate_gate)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(12):
        gate.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = gate(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        gate.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for features, labels in valid_loader:
                preds = (torch.sigmoid(gate(features)) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
        print(f"gate_epoch={epoch + 1:02d} valid_acc={correct / max(total, 1):.4f}")


def train_judge(judge: RelevanceJudge, dataset: JudgeDataset) -> None:
    train_size = int(len(dataset) * 0.85)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_judge)
    valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False, collate_fn=collate_judge)
    optimizer = torch.optim.Adam(judge.parameters(), lr=8e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(8):
        judge.train()
        for dialog_features, ad_features, tool_rows, labels in train_loader:
            optimizer.zero_grad()
            logits = judge(dialog_features, ad_features, tool_rows)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        judge.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for dialog_features, ad_features, tool_rows, labels in valid_loader:
                logits = judge(dialog_features, ad_features, tool_rows)
                total_loss += loss_fn(logits, labels).item() * labels.size(0)
                total += labels.size(0)
        print(f"judge_epoch={epoch + 1:02d} valid_loss={total_loss / max(total, 1):.4f}")


def train_orchestrator(orchestrator: SlateOrchestrator, judge: RelevanceJudge, dataset: OrchestratorDataset) -> None:
    optimizer = torch.optim.Adam(orchestrator.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = Random(11)
    for epoch in range(10):
        supervised_loss_total = 0.0
        pairwise_loss_total = 0.0
        for slate_example in dataset:
            example = slate_example.conversation
            dialog_features = dialog_to_bow(example).unsqueeze(0)
            all_scores = []
            labels = []
            for ad in slate_example.candidate_ads:
                ad_features = ad_to_vector(ad).unsqueeze(0)
                tool_tensor = tool_features(example, ad).unsqueeze(0)
                score = orchestrator(dialog_features, ad_features, tool_tensor)
                all_scores.append(score)
                labels.append(float(ad.topic == example.target_topic and example.trigger_label))
            logits = torch.cat(all_scores)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            optimizer.zero_grad()
            supervised_loss = loss_fn(logits, label_tensor)
            supervised_loss.backward()
            optimizer.step()
            supervised_loss_total += supervised_loss.item()

            if example.trigger_label:
                ranked = []
                with torch.no_grad():
                    for ad in slate_example.candidate_ads:
                        ad_features = ad_to_vector(ad).unsqueeze(0)
                        tool_tensor = tool_features(example, ad).unsqueeze(0)
                        judge_score = torch.sigmoid(judge(dialog_features, ad_features, tool_tensor)).item()
                        actor_score = orchestrator(dialog_features, ad_features, tool_tensor).item()
                        ranked.append((0.7 * actor_score + 0.3 * judge_score, ad))
                ranked.sort(key=lambda row: row[0], reverse=True)
                high_slate = [ad for _, ad in ranked[:3]]
                low_slate = [ad for _, ad in rng.sample(ranked[-6:], k=3)]
                high_reward = slate_reward(example, high_slate)
                low_reward = slate_reward(example, low_slate)
                if high_reward > low_reward:
                    optimizer.zero_grad()
                    high_score = 0.0
                    low_score = 0.0
                    for ad in high_slate:
                        high_score = high_score + orchestrator(dialog_features, ad_to_vector(ad).unsqueeze(0), tool_features(example, ad).unsqueeze(0))
                    for ad in low_slate:
                        low_score = low_score + orchestrator(dialog_features, ad_to_vector(ad).unsqueeze(0), tool_features(example, ad).unsqueeze(0))
                    pairwise_loss = torch.relu(1.0 - (high_score - low_score).squeeze())
                    pairwise_loss.backward()
                    optimizer.step()
                    pairwise_loss_total += pairwise_loss.item()
        print(
            f"orchestrator_epoch={epoch + 1:02d} supervised_loss={supervised_loss_total / len(dataset):.4f} pairwise_loss={pairwise_loss_total / max(len(dataset), 1):.4f}"
        )


def main() -> None:
    torch.manual_seed(7)
    ads = build_ads()
    conversations = build_conversations()

    gate_dataset = GateDataset(conversations)
    judge_dataset = JudgeDataset(conversations, ads)
    orchestrator_dataset = OrchestratorDataset(conversations, ads)

    dialog_dim = dialog_to_bow(conversations[0]).numel()
    ad_dim = ad_to_vector(ads[0]).numel()
    tool_dim = tool_features(conversations[0], ads[0]).numel()

    gate = OpportunityGateNet(dialog_dim)
    judge = RelevanceJudge(dialog_dim, ad_dim, tool_dim)
    orchestrator = SlateOrchestrator(dialog_dim, ad_dim, tool_dim)

    train_gate(gate, gate_dataset)
    train_judge(judge, judge_dataset)
    train_orchestrator(orchestrator, judge, orchestrator_dataset)

    out_path = Path(__file__).resolve().parent / "adsworldengine_toy.pt"
    torch.save(
        {
            "gate": gate.state_dict(),
            "judge": judge.state_dict(),
            "orchestrator": orchestrator.state_dict(),
            "dialog_dim": dialog_dim,
            "ad_dim": ad_dim,
            "tool_dim": tool_dim,
        },
        out_path,
    )
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
