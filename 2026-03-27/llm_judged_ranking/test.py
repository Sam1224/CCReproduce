import torch
from torch.utils.data import DataLoader

from dataset import ToyLLMJudgedRankingDataset, collate_fn
from model import LLMJudgedRanker


@torch.no_grad()
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ToyLLMJudgedRankingDataset(num_samples=256)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = LLMJudgedRanker().to(device)
    model.eval()

    all_logits = []
    all_labels = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (logits.sigmoid() > 0.5).float()
    acc = (preds == labels).float().mean().item()
    print(f"Accuracy={acc:.3f} (toy)")
    print("Test passed!")


if __name__ == "__main__":
    test()
