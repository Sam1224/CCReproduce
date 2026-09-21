from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data import SyntheticProductSequenceDataset, collate
from model import DSRec


def recall_at_10(logits: torch.Tensor, targets: torch.Tensor) -> float:
    topk = logits.topk(10, dim=-1).indices
    return topk.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()


def main() -> None:
    torch.manual_seed(17)
    model = DSRec()
    state = torch.load("artifacts/dsrec.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dataset = SyntheticProductSequenceDataset(size=128, seed=19)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate)
    scores = []
    with torch.no_grad():
        for batch in loader:
            scores.append(recall_at_10(model(batch.items, batch.deltas), batch.targets))
    score = sum(scores) / len(scores)
    print(f"Recall@10={score:.4f}")
    assert score >= 0.18


if __name__ == "__main__":
    main()
