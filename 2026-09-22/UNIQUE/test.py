import torch
from torch.utils.data import DataLoader
from dataset import ToyFeedDataset
from model import UNIQUE, unique_loss


def main() -> None:
    batch = next(iter(DataLoader(ToyFeedDataset(size=8), batch_size=8)))
    model = UNIQUE()
    output = model(batch)
    loss = unique_loss(output, batch)
    assert output["code_logits"].shape[0] == 8
    assert output["ctr"].shape == (8,)
    assert torch.isfinite(loss)
    print({"retrieval_logits": list(output["code_logits"].shape), "loss": round(float(loss.detach()), 4)})


if __name__ == "__main__":
    main()
