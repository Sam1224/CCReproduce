import torch
from torch.utils.data import DataLoader, random_split

from data import GroceryDataset, TOKEN_TO_ID, CATEGORIES, collate
from model import GrocLM


def precision_at_k(logits: torch.Tensor, target: torch.Tensor, top_k: int) -> float:
    selected = torch.topk(torch.sigmoid(logits), k=top_k, dim=-1).indices
    scores = []
    for row_index, row in enumerate(selected):
        hits = target[row_index, row].sum().item()
        scores.append(hits / top_k)
    return sum(scores) / len(scores)


def main() -> None:
    dataset = GroceryDataset(size=400)
    train_size = int(len(dataset) * 0.8)
    _, test_set = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(1))
    loader = DataLoader(test_set, batch_size=64, collate_fn=collate)
    model = GrocLM(vocab_size=len(TOKEN_TO_ID), category_count=len(CATEGORIES))
    try:
        model.load_state_dict(torch.load("groclm_toy.pt", map_location="cpu"))
    except FileNotFoundError:
        pass
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["history"], batch["query"], batch["rebuy_prior"])
            values.append(precision_at_k(logits, batch["target"], top_k=5))
    print({"precision_at_5": round(sum(values) / len(values), 4), "decoded_example": model.constrained_decode(logits[:1], 5)[0]})


if __name__ == "__main__":
    main()
