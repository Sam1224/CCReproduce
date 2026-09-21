from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PersonaIEDataset
from model import SelfMetaEvolve


def main() -> None:
    torch.manual_seed(29)
    dataset = PersonaIEDataset()
    model = SelfMetaEvolve(num_roles=len(dataset.role_to_id), num_fields=len(dataset.field_to_id))
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(6):
        total = 0.0
        successes = []
        for batch in DataLoader(dataset, batch_size=32, shuffle=True):
            role_id = batch["role_id"]
            target_id = batch["target_id"]
            feedback = model.role_embedding(role_id).detach() * 0.1
            logits = model(role_id, feedback)
            loss = loss_fn(logits, target_id)
            opt.zero_grad()
            loss.backward()
            opt.step()
            successes.append(feedback.mean(dim=0))
            total += float(loss.item())
        model.evolve_meta_prompt(torch.stack(successes))
        history.append(total)
        print(f"epoch={epoch + 1} objective={total:.4f}")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/self_meta_evolve.pt")
    Path("artifacts/metrics.json").write_text(json.dumps({"objective": history[-1]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
