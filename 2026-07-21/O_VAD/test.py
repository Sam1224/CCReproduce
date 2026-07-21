import argparse

import torch

from data import ANOMALY_TYPES, make_loaders
from model import OvadToy, build_report, frame_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/o_vad.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = make_loaders(batch_size=16)
    model = OvadToy().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    video_ok = 0
    object_ok = 0
    object_total = 0
    iou = 0.0
    example_report = None
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            video_ok += ((torch.sigmoid(out["video_logits"]) > 0.5).float() == batch["video_label"]).sum().item()
            valid = batch["anomaly_object"] >= 0
            if valid.any():
                pred_object = out["object_logits"].argmax(dim=-1)
                object_ok += (pred_object[valid] == batch["anomaly_object"][valid]).sum().item()
                object_total += valid.sum().item()
            iou += frame_iou(out["frame_logits"], batch["frame_mask"])
            if example_report is None:
                report = build_report(out, index=0)
                report["anomaly_type"] = ANOMALY_TYPES[report["anomaly_type_id"]]
                example_report = report

    print({
        "video_accuracy": video_ok / max(len(loader.dataset), 1),
        "object_accuracy": object_ok / max(object_total, 1),
        "frame_iou": iou / max(len(loader), 1),
        "saved_metrics": ckpt.get("metrics", {}),
        "example_report": example_report,
    })


if __name__ == "__main__":
    main()
