
import torch
from model import Agentic_llm_judgeModel
from dataset import ToyDataset, collate_fn

@torch.no_grad()
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Agentic_llm_judgeModel().to(device)
    model.eval()
    batch = collate_fn([ToyDataset()[0]])
    batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
    logits = model(batch)
    print("Test passed!")

if __name__ == "__main__":
    test()
