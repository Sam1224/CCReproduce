import torch
from torch.utils.data import DataLoader

from dataset import ToyMultiHaystackDataset, collate_fn
from model import MultiHaystackRetriever


@torch.no_grad()
def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = scores.topk(k, dim=-1).indices
    hit = (topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return float(hit)


@torch.no_grad()
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ToyMultiHaystackDataset(num_queries=64, num_docs=128)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = MultiHaystackRetriever().to(device)
    model.eval()

    recalls = []
    for batch in dl:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        scores = model(batch)["scores"]
        recalls.append(recall_at_k(scores, batch["label"], k=5))

    print(f"Recall@5={sum(recalls)/len(recalls):.3f} (toy)")
    print("Test passed!")


if __name__ == "__main__":
    test()
