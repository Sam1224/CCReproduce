from __future__ import annotations

import argparse

import torch

from dataset import SimpleTokenizer, ToyProductCatalog
from model import Moon3ToyModel


@torch.no_grad()
def recall_at_k(sim: torch.Tensor, k: int) -> float:
    # sim: [N, N] similarity between queries and catalog items
    # ground truth is diagonal.
    topk = sim.topk(k, dim=1).indices
    gt = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    hit = (topk == gt).any(dim=1).float().mean().item()
    return float(hit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    text_tok = SimpleTokenizer(ckpt["text_vocab"])
    attr_tok = SimpleTokenizer(ckpt["attr_vocab"])

    model = Moon3ToyModel(
        text_vocab_size=len(text_tok.vocab),
        attr_vocab_size=len(attr_tok.vocab),
        text_pad_id=ckpt["text_pad_id"],
        attr_pad_id=ckpt["attr_pad_id"],
        image_size=args.image_size,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(args.device)
    model.eval()

    catalog = ToyProductCatalog(seed=13)

    # Build catalog embeddings.
    images = torch.stack([catalog.image_tensor(p.product_id, size=args.image_size) for p in catalog.products]).to(args.device)
    text = torch.stack([text_tok.encode(p.title, max_len=16) for p in catalog.products]).to(args.device)

    emb = model.encode(images, text)  # [N, D]
    sim = emb @ emb.t()

    print(f"R@1  = {recall_at_k(sim, 1):.4f}")
    print(f"R@5  = {recall_at_k(sim, 5):.4f}")
    print(f"R@10 = {recall_at_k(sim, 10):.4f}")


if __name__ == "__main__":
    main()
