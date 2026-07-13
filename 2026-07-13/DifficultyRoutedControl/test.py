import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyServiceSessionDataset
from model import DifficultyRouter, WritePlanVerifier, difficulty_routed_control, build_verifier_features


def verifier_pair_accuracy(loader, verifier, device):
    verifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            conversation_vec = batch["conversation_vec"].to(device)
            action_id = batch["action_id"].to(device)
            slot_vectors = batch["slot_vectors"].to(device)
            slot_mask = batch["slot_mask"].to(device)
            gold_slot = batch["gold_slot"].to(device)
            for slot_index in range(slot_vectors.shape[1]):
                valid = slot_mask[:, slot_index] > 0.5
                if valid.sum() == 0:
                    continue
                logits = verifier(build_verifier_features(conversation_vec[valid], slot_vectors[valid, slot_index, :], action_id[valid])).squeeze(-1)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                labels = (gold_slot[valid] == slot_index).long()
                correct += (preds == labels).sum().item()
                total += labels.numel()
    return correct / max(1, total)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    dataset = ToyServiceSessionDataset(samples=args.samples, split="test", vocab=checkpoint["vocab"])
    loader = DataLoader(dataset, batch_size=args.batch_size)

    hidden_dim = checkpoint.get("hidden_dim", 128)
    router = DifficultyRouter(input_dim=dataset.vector_size, hidden_dim=hidden_dim).to(device)
    verifier = WritePlanVerifier(input_dim=dataset.vector_size * 2 + 6, hidden_dim=hidden_dim).to(device)
    router.load_state_dict(checkpoint["router"])
    verifier.load_state_dict(checkpoint["verifier"])
    router.eval()
    verifier.eval()

    router_correct = 0
    total = 0
    baseline_success = 0
    controlled_success = 0
    complex_total = 0
    baseline_complex_success = 0
    controlled_complex_success = 0
    routed_fraction = 0

    with torch.no_grad():
        for batch in loader:
            gpu_batch = {key: value.to(device) for key, value in batch.items()}
            outputs = difficulty_routed_control(gpu_batch, router, verifier, threshold=args.threshold)
            router_pred = outputs["router_logits"].argmax(dim=1)
            router_correct += (router_pred == gpu_batch["difficulty"]).sum().item()
            total += gpu_batch["difficulty"].shape[0]
            baseline_success += (outputs["baseline_choice"] == gpu_batch["gold_slot"]).sum().item()
            controlled_success += (outputs["chosen_slot"] == gpu_batch["gold_slot"]).sum().item()
            routed_fraction += outputs["escalated"].float().sum().item()
            complex_mask = gpu_batch["difficulty"] == 1
            complex_total += complex_mask.sum().item()
            if complex_mask.any():
                baseline_complex_success += (outputs["baseline_choice"][complex_mask] == gpu_batch["gold_slot"][complex_mask]).sum().item()
                controlled_complex_success += (outputs["chosen_slot"][complex_mask] == gpu_batch["gold_slot"][complex_mask]).sum().item()

    metrics = {
        "router_acc": round(router_correct / max(1, total), 4),
        "baseline_success": round(baseline_success / max(1, total), 4),
        "controlled_success": round(controlled_success / max(1, total), 4),
        "baseline_complex_success": round(baseline_complex_success / max(1, complex_total), 4),
        "controlled_complex_success": round(controlled_complex_success / max(1, complex_total), 4),
        "routed_fraction": round(routed_fraction / max(1, total), 4),
        "verifier_pair_acc": round(verifier_pair_accuracy(loader, verifier, device), 4),
    }
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/difficulty_routed_control.pt")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
