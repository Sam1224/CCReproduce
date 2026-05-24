from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SimpleTokenizer, ToyMOONTripletDataset, ToyProductCatalog, collate_triplets
from model import Moon3ToyModel


def contrastive_pair_loss(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """A simple (q, pos, neg) InfoNCE variant."""

    pos = (q * p).sum(dim=-1) / tau
    neg = (q * n).sum(dim=-1) / tau

    logits = torch.stack([pos, neg], dim=1)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def rl_stub_loss(*_args, **_kwargs) -> torch.Tensor:
    """Placeholder for the paper's GRPO-like optimization.

    In MOON3.0, RL is used to explore better attribute-deconstruction reasoning strategies.

    A high-level sketch (pseudocode):

    ```
    for batch in dataloader:
        # 1) sample G reasoning generations (attributes / CoT)
        generations = sample_attr_sequences(model, G)

        # 2) compute rewards per generation
        #   - retrieval accuracy reward: does the embedding retrieve the positive?
        #   - quality / length reward: penalize too long / low-quality generations
        rewards = compute_reward(generations)

        # 3) policy optimization (e.g., GRPO / PPO)
        rl_loss = grpo(policy=model.attr_head, rewards=rewards)
    ```

    This toy reproduction does NOT implement GRPO because it requires careful sampling,
    off-policy correction, and stable reward shaping.
    """

    return torch.tensor(0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--max_text_len", type=int, default=16)
    parser.add_argument("--max_attr_len", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda_attr", type=float, default=1.0)
    parser.add_argument("--lambda_rl", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="checkpoints/moon3_0.pt")
    args = parser.parse_args()

    torch.manual_seed(0)

    catalog = ToyProductCatalog(seed=13)
    dataset = ToyMOONTripletDataset(
        catalog=catalog,
        num_samples=args.num_samples,
        seed=7,
        image_size=args.image_size,
    )

    # Build tokenizers.
    text_tok = SimpleTokenizer.build([p.title for p in catalog.products])

    # Attributes are structured; add explicit prefixes to vocab so the model can copy them.
    attr_texts = [ToyMOONTripletDataset.attributes_to_text(p.attributes) for p in catalog.products]
    extra = ["category:", "color:", "material:", "style:"]
    attr_tok = SimpleTokenizer.build(attr_texts, extra_tokens=extra)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_triplets(
            b,
            text_tokenizer=text_tok,
            attr_tokenizer=attr_tok,
            max_text_len=args.max_text_len,
            max_attr_len=args.max_attr_len,
        ),
    )

    model = Moon3ToyModel(
        text_vocab_size=len(text_tok.vocab),
        attr_vocab_size=len(attr_tok.vocab),
        text_pad_id=text_tok.pad_id,
        attr_pad_id=attr_tok.pad_id,
        image_size=args.image_size,
    ).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            q_img = batch["q_image"].to(args.device)
            p_img = batch["p_image"].to(args.device)
            n_img = batch["n_image"].to(args.device)
            q_txt = batch["q_text"].to(args.device)
            p_txt = batch["p_text"].to(args.device)
            n_txt = batch["n_text"].to(args.device)
            q_attr = batch["q_attr"].to(args.device)

            q_out = model(q_img, q_txt, q_attr)
            p_emb = model.encode(p_img, p_txt)
            n_emb = model.encode(n_img, n_txt)

            info_nce = contrastive_pair_loss(q_out.embedding, p_emb, n_emb, tau=args.tau)
            rl_loss = rl_stub_loss()

            loss = info_nce + args.lambda_attr * q_out.attr_loss + args.lambda_rl * rl_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total += float(loss.detach())

        avg = total / max(1, len(loader))
        print(f"epoch={epoch:03d} loss={avg:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "text_vocab": text_tok.vocab,
        "attr_vocab": attr_tok.vocab,
        "text_pad_id": text_tok.pad_id,
        "attr_pad_id": attr_tok.pad_id,
    }
    torch.save(ckpt, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
