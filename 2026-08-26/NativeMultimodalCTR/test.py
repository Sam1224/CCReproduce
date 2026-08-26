from __future__ import annotations

import json
from pathlib import Path

import torch

from data import binary_metrics, build_dataloaders, make_batch
from model import NativeMultimodalCTR


def main() -> None:
    seed = 26
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, _, _, _, test_dl = build_dataloaders(seed=seed)

    model = NativeMultimodalCTR(
        num_users=world.user_pref.shape[0],
        num_categories=int(world.category.max()) + 1,
        text_dim=world.text_feat.shape[1],
        image_dim=world.image_feat.shape[1],
    )
    ckpt_path = Path(__file__).resolve().parent / "artifacts" / "native_multimodal_ctr.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    logits, labels, gates = [], [], []
    with torch.no_grad():
        for raw in test_dl:
            batch = {k: v.to(device) for k, v in make_batch(world, raw).items()}
            out = model(
                user_id=batch["user_id"],
                text=batch["text"],
                image=batch["image"],
                category=batch["category"],
                price=batch["price"],
            )
            logits.append(out.logits.cpu())
            labels.append(batch["label"].cpu())
            gates.append(out.gate.cpu())

    metrics = binary_metrics(torch.cat(logits), torch.cat(labels))
    metrics["mean_text_gate"] = float(torch.cat(gates).mean())
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
