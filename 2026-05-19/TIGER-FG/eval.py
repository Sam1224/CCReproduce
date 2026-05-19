import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiger_fg.dataset import (
    ToyIMMRGallery,
    ToyIMMRItemDataset,
    ToyIMMRQueryDataset,
    collate_item,
    collate_query,
)
from tiger_fg.metrics import recall_at_k
from tiger_fg.model import TIGERFG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_items", type=int, default=400)
    parser.add_argument("--num_queries", type=int, default=200)
    args = parser.parse_args()

    model = TIGERFG(embedding_dim=256).to(args.device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    gallery = ToyIMMRGallery(num_items=args.num_items)
    item_loader = DataLoader(
        ToyIMMRItemDataset(gallery),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_item,
    )
    query_loader = DataLoader(
        ToyIMMRQueryDataset(gallery, num_queries=args.num_queries),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_query,
    )

    item_embs = []
    with torch.no_grad():
        for batch in tqdm(item_loader, desc="encode items"):
            out = model.encode_item(
                item_images=batch["item_image"].to(args.device),
                item_text=batch["item_text"].to(args.device),
            )
            item_embs.append(out["item_emb"].cpu())

    item_emb = torch.cat(item_embs, dim=0)

    query_embs = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="encode queries"):
            query_embs.append(model.encode_query(batch["query_image"].to(args.device)).cpu())
            targets.append(batch["target"].cpu())

    query_emb = torch.cat(query_embs, dim=0)
    targets = torch.cat(targets, dim=0)

    sims = query_emb @ item_emb.T

    r1 = recall_at_k(sims, targets, k=1)
    r5 = recall_at_k(sims, targets, k=5)

    print({"Recall@1": r1, "Recall@5": r5})


if __name__ == "__main__":
    main()
