from __future__ import annotations

import torch

from data import build_dataloaders
from model import ComplianceAuditor, MultiVectorRetriever


def main() -> None:
    pages, _, _, test_loader = build_dataloaders()
    retriever = MultiVectorRetriever()
    retriever.load_state_dict(torch.load("plansightrag_retriever.pt", map_location="cpu"))
    auditor = ComplianceAuditor()
    retriever.eval()
    hits = verdict_correct = total = 0
    rule_values = torch.tensor([p.rule_value for p in pages])
    with torch.no_grad():
        for batch in test_loader:
            out = retriever(batch["query_vec"], pages, topk=5)
            top1 = out.topk_ids[:, 0]
            hits += int((out.topk_ids == batch["gold_page"].unsqueeze(1)).any(dim=1).sum())
            retrieved_rule = rule_values[top1]
            pred = auditor(retrieved_rule, batch["proposed_value"]).argmax(dim=-1)
            gold = (rule_values[batch["gold_page"]] >= batch["proposed_value"] + auditor.margin.abs()).long()
            verdict_correct += int((pred == gold).sum())
            total += int(batch["gold_page"].numel())
    print(f"recall@5={hits / total:.3f}")
    print(f"toy_verdict_accuracy={verdict_correct / total:.3f}")


if __name__ == "__main__":
    main()
