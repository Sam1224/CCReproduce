"""Evaluation script for TGQ-Former.

Builds gallery embeddings from all dataset items, then evaluates H@10 and H@100.

Usage:
    python eval.py --ckpt runs/tgqformer/model.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tgqformer import TGQFormer, ECommerceDataset
from tgqformer.metrics import hit_rate_at_k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/tgqformer/model.pt")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--num_products", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def build_gallery(model, dataset, device, batch_size=128):
    """Return (all_embs, all_pids) tensors."""
    from torch.utils.data import DataLoader as DL

    loader = DL(dataset, batch_size=batch_size, shuffle=False)
    all_embs, all_pids = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            emb = model.encode_item(
                batch["q_image"].to(device), batch["q_text"].to(device)
            )
            all_embs.append(emb.cpu())
            all_pids.append(batch["product_id"] if "product_id" in batch
                            else torch.zeros(emb.size(0), dtype=torch.long))
    return torch.cat(all_embs, 0), torch.cat(all_pids, 0)


class GalleryDataset(torch.utils.data.Dataset):
    """Wraps ECommerceDataset to expose per-item image/text/pid without pairs."""
    def __init__(self, base):
        self.items = base.items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "q_image": it["image"],
            "q_text": it["text_tokens"],
            "product_id": torch.tensor(it["product_id"]),
        }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TGQFormer(d_model=args.d_model, emb_dim=args.emb_dim)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    base_ds = ECommerceDataset(num_products=args.num_products)
    gallery_ds = GalleryDataset(base_ds)

    embs, pids = build_gallery(model, gallery_ds, device, args.batch_size)
    print(f"Gallery size: {embs.size(0)} items, {args.num_products} products")

    h10 = hit_rate_at_k(embs, embs, pids, pids, k=10)
    h100 = hit_rate_at_k(embs, embs, pids, pids, k=100)
    print(f"H@10  = {h10:.4f}")
    print(f"H@100 = {h100:.4f}")


if __name__ == "__main__":
    main()
