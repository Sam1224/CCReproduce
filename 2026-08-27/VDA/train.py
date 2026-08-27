from __future__ import annotations

import torch
import torch.nn.functional as F

from data import build_continual_loaders
from model import VDAModel, vc_ot_loss, visually_modulated_loss


def evaluate(model, loaders) -> float:
    model.eval()
    accs = []
    with torch.no_grad():
        for loader in loaders:
            correct = total = 0
            for batch in loader:
                pred = model(batch).logits.argmax(dim=-1)
                correct += int((pred == batch["labels"]).sum())
                total += int(batch["labels"].numel())
            accs.append(correct / max(total, 1))
    return sum(accs) / len(accs)


def main() -> None:
    torch.manual_seed(23)
    train_loaders, test_loaders = build_continual_loaders()
    model = VDAModel()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3)
    replay = []
    reference_vd = []
    for task_id, loader in enumerate(train_loaders):
        for epoch in range(2):
            model.train()
            for batch in loader:
                out = model(batch)
                loss = F.cross_entropy(out.logits, batch["labels"]) + 0.15 * visually_modulated_loss(out, batch)
                if replay:
                    memory = replay[-1]
                    ref = reference_vd[-1]
                    mem_out = model(memory)
                    loss = loss + 0.10 * vc_ot_loss(mem_out.visual_dependence, ref)
                optim.zero_grad()
                loss.backward()
                optim.step()
        first_batch = next(iter(loader))
        with torch.no_grad():
            reference_vd.append(model(first_batch).visual_dependence.detach())
            replay.append(first_batch)
        print(f"after_task={task_id} avg_acc={evaluate(model, test_loaders[: task_id + 1]):.3f}")
    torch.save(model.state_dict(), "vda_toy.pt")
    print(f"final_avg_acc={evaluate(model, test_loaders):.3f}")


if __name__ == "__main__":
    main()
