import argparse

import torch

from data import make_loaders
from model import GMOE2DITToy, mask_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gmo_e2dit.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = make_loaders(batch_size=32)
    model = GMOE2DITToy().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    op_ok = 0
    count = 0
    src_iou = 0.0
    tgt_iou = 0.0
    edit_l1 = 0.0
    reflection_ok = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            op_ok += (out["op_logits"].argmax(-1) == batch["op_id"]).sum().item()
            count += batch["op_id"].numel()
            src_iou += mask_iou(out["source_logits"], batch["source_mask"])
            tgt_iou += mask_iou(out["target_logits"], batch["target_mask"])
            edit_l1 += torch.nn.functional.l1_loss(out["edited"], batch["target"]).item()
            reflection_ok += ((torch.sigmoid(out["success_logits"]) > 0.5).float() == batch["success"]).sum().item()
    n = max(len(loader), 1)
    print({
        "operation_accuracy": op_ok / max(count, 1),
        "source_mask_iou": src_iou / n,
        "target_mask_iou": tgt_iou / n,
        "edit_l1": edit_l1 / n,
        "reflection_accuracy": reflection_ok / max(count, 1),
    })


if __name__ == "__main__":
    main()
