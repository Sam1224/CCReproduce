import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyServiceSessionDataset
from model import DifficultyRouter, WritePlanVerifier, build_verifier_features, difficulty_routed_control


def train_router(router, loader, optimizer, device):
    router.train()
    total_loss = 0.0
    for batch in loader:
        conversation_vec = batch["conversation_vec"].to(device)
        labels = batch["difficulty"].to(device)
        logits = router(conversation_vec)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def train_verifier(verifier, loader, optimizer, device):
    verifier.train()
    total_loss = 0.0
    for batch in loader:
        conversation_vec = batch["conversation_vec"].to(device)
        action_id = batch["action_id"].to(device)
        slot_vectors = batch["slot_vectors"].to(device)
        slot_mask = batch["slot_mask"].to(device)
        gold_slot = batch["gold_slot"].to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for slot_index in range(slot_vectors.shape[1]):
            valid = slot_mask[:, slot_index] > 0.5
            if valid.sum() == 0:
                continue
            features = build_verifier_features(conversation_vec[valid], slot_vectors[valid, slot_index, :], action_id[valid])
            logits = verifier(features).squeeze(-1)
            labels = (gold_slot[valid] == slot_index).float()
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(dev_loader, router, verifier, device):
    router.eval()
    verifier.eval()
    router_correct = 0
    total = 0
    baseline_success = 0
    controlled_success = 0
    complex_total = 0
    complex_controlled = 0
    with torch.no_grad():
        for batch in dev_loader:
            gpu_batch = {key: value.to(device) for key, value in batch.items()}
            outputs = difficulty_routed_control(gpu_batch, router, verifier)
            router_pred = outputs["router_logits"].argmax(dim=1)
            router_correct += (router_pred == gpu_batch["difficulty"]).sum().item()
            total += gpu_batch["difficulty"].shape[0]
            baseline_success += (outputs["baseline_choice"] == gpu_batch["gold_slot"]).sum().item()
            controlled_success += (outputs["chosen_slot"] == gpu_batch["gold_slot"]).sum().item()
            complex_mask = gpu_batch["difficulty"] == 1
            complex_total += complex_mask.sum().item()
            if complex_mask.any():
                complex_controlled += (outputs["chosen_slot"][complex_mask] == gpu_batch["gold_slot"][complex_mask]).sum().item()
    return {
        "router_acc": router_correct / max(1, total),
        "baseline_success": baseline_success / max(1, total),
        "controlled_success": controlled_success / max(1, total),
        "complex_controlled": complex_controlled / max(1, complex_total),
    }


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_dataset = ToyServiceSessionDataset(samples=args.samples, split="train")
    dev_dataset = ToyServiceSessionDataset(samples=max(128, args.samples // 2), split="test", vocab=train_dataset.vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    router = DifficultyRouter(input_dim=train_dataset.vector_size, hidden_dim=args.hidden_dim).to(device)
    verifier = WritePlanVerifier(input_dim=train_dataset.vector_size * 2 + 6, hidden_dim=args.hidden_dim).to(device)
    router_optim = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    verifier_optim = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        router_loss = train_router(router, train_loader, router_optim, device)
        verifier_loss = train_verifier(verifier, train_loader, verifier_optim, device)
        metrics = evaluate(dev_loader, router, verifier, device)
        print(
            f"epoch={epoch + 1} router_loss={router_loss:.4f} verifier_loss={verifier_loss:.4f} "
            f"router_acc={metrics['router_acc']:.3f} controlled_success={metrics['controlled_success']:.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "difficulty_routed_control.pt"
    torch.save(
        {
            "router": router.state_dict(),
            "verifier": verifier.state_dict(),
            "vocab": train_dataset.vocab,
            "hidden_dim": args.hidden_dim,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
