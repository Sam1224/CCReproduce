"""LiveStarPro — Evaluation & Streaming Demo"""

import argparse
import torch

from data import generate_toy_stream_data, build_dataloaders
from model import LiveStarPro


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    timing_abs_err = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        respond_at = batch["respond_at"].to(device)
        response_class = batch["response_class"].to(device)
        B, T, C, H, W = frames.shape

        out = model.forward_training(frames, respond_at, response_class)
        pred_respond = out["sved_scores"].argmax(dim=-1)
        timing_abs_err += (pred_respond - respond_at).float().abs().mean().item()

        frame_embs = model.frame_encoder(frames.view(B * T, C, H, W)).view(B, T, -1)
        ctx = model.temporal_transformer(frame_embs)
        respond_ctx = ctx[torch.arange(B), respond_at.clamp(0, T - 1)]
        preds = model.response_head(respond_ctx).argmax(dim=-1)
        correct += (preds == response_class).sum().item()
        total += B

    n = len(loader)
    return {
        "response_accuracy": correct / total,
        "mean_timing_error": timing_abs_err / n,
    }


def streaming_demo(model, device, n_frames: int = 40):
    """Simulate real-time streaming inference with TSHM + SVeD."""
    model.eval()
    kv_cache = []
    memory_nodes = []
    responses = []

    print("\n=== Streaming Inference Demo ===")
    with torch.no_grad():
        for i in range(n_frames):
            frame = torch.randn(1, 3, 32, 32).to(device)
            result = model.stream_inference(frame, kv_cache, memory_nodes)
            status = "RESPOND" if result["should_respond"].item() else "silent"
            print(f"  Frame {i:3d} | SVeD score: {result['score'].item():.3f} | {status} "
                  f"| KV cache: {len(kv_cache):2d} | Memory nodes: {len(memory_nodes):2d}")
            if result["should_respond"].item() and "response" in result:
                responses.append(result["response"].item())

    print(f"\nTotal responses generated: {len(responses)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="liveStarPro_best.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_toy_stream_data()
    _, val_loader = build_dataloaders(data, batch_size=args.batch_size)

    model = LiveStarPro(
        embed_dim=128, n_heads=4, n_layers=2, chunk_size=4,
        n_response_classes=data["n_response_classes"]
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded {args.ckpt}")
    except FileNotFoundError:
        print("No checkpoint; evaluating random init.")

    results = evaluate(model, val_loader, device)
    print("\n=== LiveStarPro Evaluation ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    if args.demo:
        streaming_demo(model, device)


if __name__ == "__main__":
    main()
