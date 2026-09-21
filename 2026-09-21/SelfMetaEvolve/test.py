from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data import PersonaIEDataset
from model import SelfMetaEvolve


def main() -> None:
    dataset = PersonaIEDataset(size=96, seed=31)
    model = SelfMetaEvolve(num_roles=len(dataset.role_to_id), num_fields=len(dataset.field_to_id))
    model.load_state_dict(torch.load("artifacts/self_meta_evolve.pt", map_location="cpu"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=48):
            logits = model(batch["role_id"])
            correct += int((logits.argmax(dim=-1) == batch["target_id"]).sum().item())
            total += int(batch["target_id"].numel())
    acc = correct / total
    print(f"persona-field success={acc:.4f}")
    assert acc >= 0.20


if __name__ == "__main__":
    main()
