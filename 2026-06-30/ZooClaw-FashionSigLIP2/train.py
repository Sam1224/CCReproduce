from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data import ToyFashionRetrievalDataset, build_default_vocab, collate, split_indices
from model import DualEncoder, DualEncoderConfig, contrastive_loss, distill_kl


def teacher_logits_from_embeddings(t_img: torch.Tensor, t_txt: torch.Tensor, scale: float = 30.0) -> torch.Tensor:
    # embeddings are already normalized
    return scale * (t_img @ t_txt.t())


def train_epoch(model: DualEncoder, loader: DataLoader, opt: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["short_ids"].to(device)
        attn = batch["short_attn"].to(device)
        t_img = batch["teacher_image_emb"].to(device)
        t_txt = batch["teacher_short_emb"].to(device)

        logits = model(images, input_ids, attn)
        loss_ctr = contrastive_loss(logits)
        loss_kd = distill_kl(logits, teacher_logits_from_embeddings(t_img, t_txt), temperature=1.5)
        loss = loss_ctr + 0.7 * loss_kd

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


@torch.no_grad()
def eval_loss(model: DualEncoder, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["short_ids"].to(device)
        attn = batch["short_attn"].to(device)
        t_img = batch["teacher_image_emb"].to(device)
        t_txt = batch["teacher_short_emb"].to(device)

        logits = model(images, input_ids, attn)
        loss = contrastive_loss(logits) + 0.7 * distill_kl(
            logits, teacher_logits_from_embeddings(t_img, t_txt), temperature=1.5
        )
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def main() -> None:
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = build_default_vocab()

    # Stage 0: "foundation" pre-training on a broad distribution
    base_ds = ToyFashionRetrievalDataset(num_items=1200, seed=1, ood=False)
    train_idx, val_idx = split_indices(len(base_ds), train_ratio=0.85, seed=0)

    cfg = DualEncoderConfig(embed_dim=128, vocab_size=len(vocab.itos), max_len=32)
    base = DualEncoder(cfg).to(device)

    train_loader = DataLoader(
        Subset(base_ds, train_idx),
        batch_size=64,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate(b, vocab, max_len=cfg.max_len),
    )
    val_loader = DataLoader(
        Subset(base_ds, val_idx),
        batch_size=128,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate(b, vocab, max_len=cfg.max_len),
    )

    opt = torch.optim.AdamW(base.parameters(), lr=2e-3, weight_decay=1e-3)

    best = None
    best_state = None
    for epoch in range(1, 6):
        tr = train_epoch(base, train_loader, opt, device=device)
        va = eval_loss(base, val_loader, device=device)
        if best is None or va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        print(f"[foundation] epoch={epoch:02d} train={tr:.4f} val={va:.4f}")

    if best_state is None:
        best_state = base.state_dict()

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = ckpt_dir / "base.pt"
    torch.save({"cfg": asdict(cfg), "vocab": vocab.itos, "state_dict": best_state}, base_ckpt)
    print(f"saved base: {base_ckpt}")

    # Stage 1: "distilled full fine-tuning" on curated in-domain data
    finetune_ds = ToyFashionRetrievalDataset(num_items=800, seed=7, ood=False)
    ft_train, ft_val = split_indices(len(finetune_ds), train_ratio=0.85, seed=0)

    ft_train_loader = DataLoader(
        Subset(finetune_ds, ft_train),
        batch_size=64,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate(b, vocab, max_len=cfg.max_len),
    )
    ft_val_loader = DataLoader(
        Subset(finetune_ds, ft_val),
        batch_size=128,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate(b, vocab, max_len=cfg.max_len),
    )

    finetuned = DualEncoder(cfg).to(device)
    finetuned.load_state_dict(best_state)

    opt2 = torch.optim.AdamW(finetuned.parameters(), lr=8e-4, weight_decay=1e-3)

    best2 = None
    best2_state = None
    for epoch in range(1, 6):
        tr = train_epoch(finetuned, ft_train_loader, opt2, device=device)
        va = eval_loss(finetuned, ft_val_loader, device=device)
        if best2 is None or va < best2:
            best2 = va
            best2_state = {k: v.detach().cpu().clone() for k, v in finetuned.state_dict().items()}
        print(f"[finetune+distill] epoch={epoch:02d} train={tr:.4f} val={va:.4f}")

    if best2_state is None:
        best2_state = finetuned.state_dict()

    ft_ckpt = ckpt_dir / "finetuned.pt"
    torch.save({"cfg": asdict(cfg), "vocab": vocab.itos, "state_dict": best2_state}, ft_ckpt)
    print(f"saved finetuned: {ft_ckpt}")


if __name__ == "__main__":
    main()
