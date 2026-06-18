"""LiveStarPro — Training Script"""

import argparse
import torch
import torch.optim as optim

from data import generate_toy_stream_data, build_dataloaders
from model import LiveStarPro


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_resp = total_timing = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        respond_at = batch["respond_at"].to(device)
        response_class = batch["response_class"].to(device)

        out = model.forward_training(frames, respond_at, response_class)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_resp += out["L_response"]
        total_timing += out["L_timing"]

    n = len(loader)
    return total_loss / n, total_resp / n, total_timing / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    timing_err = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        respond_at = batch["respond_at"].to(device)
        response_class = batch["response_class"].to(device)

        out = model.forward_training(frames, respond_at, response_class)

        # Accuracy: predicted respond frame vs ground truth
        pred_respond = out["sved_scores"].argmax(dim=-1)
        timing_err += (pred_respond - respond_at).float().abs().mean().item()

        B, T, C, H, W = frames.shape
        respond_ctx_idx = respond_at.clamp(0, T - 1)
        frame_embs = model.frame_encoder(frames.view(B * T, C, H, W)).view(B, T, -1)
        ctx = model.temporal_transformer(frame_embs)
        respond_ctx = ctx[torch.arange(B), respond_ctx_idx]
        preds = model.response_head(respond_ctx).argmax(dim=-1)
        correct += (preds == response_class).sum().item()
        total += B

    n = len(loader)
    return correct / total, timing_err / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = generate_toy_stream_data()
    train_loader, val_loader = build_dataloaders(data, batch_size=args.batch_size)

    model = LiveStarPro(
        embed_dim=args.embed_dim,
        n_heads=4,
        n_layers=2,
        chunk_size=4,
        n_response_classes=data["n_response_classes"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0
    print(f"\nTraining LiveStarPro for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss, l_resp, l_timing = train_epoch(model, train_loader, optimizer, device)

        if epoch % 5 == 0 or epoch == 1:
            acc, t_err = evaluate(model, val_loader, device)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "liveStarPro_best.pt")
            print(
                f"Epoch {epoch:3d} | Loss={loss:.4f} L_resp={l_resp:.4f} "
                f"L_timing={l_timing:.4f} | Acc={acc:.4f} TimingErr={t_err:.2f} (best_acc={best_acc:.4f})"
            )

    print(f"\nTraining complete. Best Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
