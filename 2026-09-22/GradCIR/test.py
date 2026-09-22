import torch
from torch.utils.data import DataLoader
from dataset import ToyGradCIRDataset
from model import GradCIR, ndcg_at_k


def main() -> None:
    dataset = ToyGradCIRDataset(size=16)
    batch = next(iter(DataLoader(dataset, batch_size=16)))
    model = GradCIR()
    with torch.no_grad():
        scores = model(batch).diag()
        metric = ndcg_at_k(scores, batch["relevance"], k=10)
    assert scores.shape == (16,)
    assert 0.0 <= metric <= 1.0
    print({"scores_shape": list(scores.shape), "ndcg_at_10": round(metric, 4)})


if __name__ == "__main__":
    main()
