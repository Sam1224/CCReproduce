import argparse
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ssr_model import SSRConfig, SSRToyModel
from src.toy_data import ToyQueryDataset, collate_queries, load_docs, set_seed


def _pad_docs(docs, doc_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch_docs = [docs[int(i)] for i in doc_ids.tolist()]
    max_len = max(len(d) for d in batch_docs)
    doc_tensor = torch.zeros((len(batch_docs), max_len), dtype=torch.long)
    doc_mask = torch.zeros((len(batch_docs), max_len), dtype=torch.bool)
    for i, d in enumerate(batch_docs):
        doc_tensor[i, : len(d)] = torch.tensor(d, dtype=torch.long)
        doc_mask[i, : len(d)] = True
    return {"doc_ids": doc_tensor, "doc_mask": doc_mask}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.2, help="contrastive loss weight")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)

    docs = load_docs(args.data_dir)
    dataset = ToyQueryDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_queries)

    cfg = SSRConfig(vocab_size=2000)
    model = SSRToyModel(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_steps = 0
        for batch in loader:
            query_ids = batch["query_ids"].to(device)
            query_mask = batch["query_mask"].to(device)
            pos_doc_ids = batch["pos_doc_ids"].to(device)

            doc_batch = _pad_docs(docs, pos_doc_ids)
            doc_ids = doc_batch["doc_ids"].to(device)
            doc_mask = doc_batch["doc_mask"].to(device)

            optim.zero_grad()

            q_dense = model.encode_dense(query_ids)
            d_dense = model.encode_dense(doc_ids)
            q_hat, _, _ = model.encode_sparse(query_ids)
            d_hat, _, _ = model.encode_sparse(doc_ids)

            loss_recon = model.loss_recon(q_dense, q_hat, query_mask) + model.loss_recon(d_dense, d_hat, doc_mask)
            loss_ce = model.contrastive_ce_loss_sparse(query_ids, query_mask, doc_ids, doc_mask)
            loss = loss_recon + args.gamma * loss_ce

            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            total_steps += 1

        avg = total_loss / max(total_steps, 1)
        print(f"epoch={epoch+1} loss={avg:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ssr_toy.pt")
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, ckpt_path)
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
