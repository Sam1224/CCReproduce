import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, FINE_LABELS, PAD_ID, XuanwuToyDataset, build_vocab
from model import XuanwuVL2BToy, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate toy Xuanwu-VL-2B moderation robustness.")
    parser.add_argument("--checkpoint", type=str, default="xuanwu_vl_2b_toy.pt")
    parser.add_argument("--test-samples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def binary_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1.0)).item())


def evaluate(model: XuanwuVL2BToy, loader: DataLoader) -> dict[str, float | torch.Tensor]:
    model.eval()
    clean_correct = 0
    adv_correct = 0
    deploy_correct = 0
    total = 0
    fine_preds = []
    fine_targets = []
    weight_sum = torch.zeros(3)
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["image"].float(),
                batch["text_tokens"].long(),
                batch["ocr_tokens"].long(),
                batch["adv_ocr_tokens"].long(),
            )
            labels = batch["label"].long()
            clean_correct += int((outputs["logits"].argmax(dim=-1) == labels).sum().item())
            adv_correct += int((outputs["adv_logits"].argmax(dim=-1) == labels).sum().item())
            deploy_correct += int((outputs["deploy_logits"].argmax(dim=-1) == labels).sum().item())
            total += labels.numel()
            fine_preds.append(torch.sigmoid(outputs["fine_logits"]) > 0.5)
            fine_targets.append(batch["fine_labels"].bool())
            weight_sum += outputs["modality_weights"].mean(dim=0).cpu()
    clean_acc = clean_correct / max(total, 1)
    adv_acc = adv_correct / max(total, 1)
    return {
        "clean_acc": clean_acc,
        "adv_ocr_acc": adv_acc,
        "robust_gap": clean_acc - adv_acc,
        "deploy_acc": deploy_correct / max(total, 1),
        "fine_f1": binary_f1(torch.cat(fine_preds, dim=0), torch.cat(fine_targets, dim=0)),
        "modality_weights": weight_sum / max(len(loader), 1),
    }


def main() -> None:
    args = parse_args()
    package = torch.load(Path(args.checkpoint), map_location="cpu")
    config = package["config"]
    dataset = XuanwuToyDataset(num_samples=args.test_samples, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = XuanwuVL2BToy(
        vocab_size=config.get("vocab_size", len(build_vocab())),
        num_classes=config.get("num_classes", len(CLASS_NAMES)),
        fine_dim=config.get("fine_dim", len(FINE_LABELS)),
        hidden_dim=config.get("hidden_dim", 64),
        embed_dim=config.get("embed_dim", 48),
        pad_id=config.get("pad_id", PAD_ID),
    )
    model.load_state_dict(package["model_state"])
    metrics = evaluate(model, loader)

    weights = metrics["modality_weights"]
    print("Xuanwu_VL_2B toy evaluation")
    print("-" * 44)
    print(f"clean_acc      : {metrics['clean_acc']:.4f}")
    print(f"adv_ocr_acc    : {metrics['adv_ocr_acc']:.4f}")
    print(f"robust_gap     : {metrics['robust_gap']:.4f}")
    print(f"fine_f1        : {metrics['fine_f1']:.4f}")
    print(f"deploy_acc     : {metrics['deploy_acc']:.4f}")
    print(f"modality_weight: visual={weights[0]:.3f} text={weights[1]:.3f} ocr={weights[2]:.3f}")
    print(f"trainable_parameters={count_parameters(model)}")

    first = dataset[1]
    with torch.no_grad():
        demo = model(
            first["image"].unsqueeze(0).float(),
            first["text_tokens"].unsqueeze(0),
            first["ocr_tokens"].unsqueeze(0),
            first["adv_ocr_tokens"].unsqueeze(0),
        )
    clean_pred = int(demo["logits"].argmax(dim=-1).item())
    adv_pred = int(demo["adv_logits"].argmax(dim=-1).item())
    label = int(first["label"].item())
    active_fine = [name for name, value in zip(FINE_LABELS, first["fine_labels"].tolist()) if value > 0.5]
    print("\ndebug_sample")
    print(f"label={CLASS_NAMES[label]} clean_pred={CLASS_NAMES[clean_pred]} adv_pred={CLASS_NAMES[adv_pred]} fine={active_fine}")


if __name__ == "__main__":
    main()
