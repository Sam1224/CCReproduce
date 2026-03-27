
import torch
from torch.utils.data import DataLoader
from dataset import ToyDataset, collate_fn
from model import Vla_rank_consistencyModel

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ToyDataset()
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    model = Vla_rank_consistencyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(1):
        for batch in dl:
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 1000), batch["labels"].view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    train()
