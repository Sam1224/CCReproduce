import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiger_fg.dataset import ToyIMMRDataset, collate_immr
from tiger_fg.losses import tiger_fg_loss
from tiger_fg.model import TIGERFG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyIMMRDataset(size=6000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_immr,
        drop_last=True,
    )

    model = TIGERFG(embedding_dim=256).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        running = 0.0
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            out = model(
                query_images=batch["query_image"],
                item_images=batch["item_image"],
                item_text=batch["item_text"],
                gt_box_mask=batch["item_box_mask"],
            )

            loss = tiger_fg_loss(
                query_emb=out["query_emb"],
                item_emb=out["item_emb"],
                patch_attn=out["patch_attn"],
                gt_box_mask=batch["item_box_mask"],
                teacher_item_emb=out.get("teacher_item_emb"),
                temperature=0.07,
                w_contrast=1.0,
                w_spatial=0.2,
                w_simdist=0.2,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += float(loss.detach().cpu())
            pbar.set_postfix(loss=running / step)

        torch.save(model.state_dict(), output_dir / "model.pt")

    print(f"saved: {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
