import os
import json
import time
import argparse
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import data as data_lib
from model import build_model


def parse_args():
    p = argparse.ArgumentParser("CRID Generative Retrieval (toy) - train")
    p.add_argument("--id_scheme", type=str, default="crid", choices=["crid", "sid"], help="对比 CRID vs SID")

    p.add_argument("--artifact_path", type=str, default="artifacts/toy_data.pt", help="数据 artifact 路径")
    p.add_argument("--run_dir", type=str, default=None, help="输出目录（默认 runs/{id_scheme}）")
    p.add_argument(
        "--force_regen",
        action="store_true",
        help="强制重新生成数据 artifact（当你修改了数据分布/超参时建议打开）",
    )

    # data cfg
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_clusters", type=int, default=24)
    p.add_argument("--docs_per_cluster", type=int, default=20)
    p.add_argument("--train_queries_per_cluster", type=int, default=60)
    p.add_argument("--test_queries_per_cluster", type=int, default=30)
    p.add_argument("--value_alpha", type=float, default=3.0)
    p.add_argument("--value_tau", type=float, default=3.0)
    p.add_argument("--holdout_ratio", type=float, default=0.35)
    p.add_argument("--holdout_topk", type=int, default=2)

    # train
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)

    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    return p.parse_args()


def maybe_build_artifact(args) -> Dict[str, Any]:
    if os.path.exists(args.artifact_path) and (not args.force_regen):
        return data_lib.load_artifact(args.artifact_path)

    cfg = data_lib.ToyConfig(
        seed=args.seed,
        n_clusters=args.n_clusters,
        docs_per_cluster=args.docs_per_cluster,
        train_queries_per_cluster=args.train_queries_per_cluster,
        test_queries_per_cluster=args.test_queries_per_cluster,
        value_alpha=args.value_alpha,
        value_tau=args.value_tau,
        holdout_ratio=args.holdout_ratio,
        holdout_topk=args.holdout_topk,
    )
    artifact = data_lib.generate_toy_data(cfg)
    data_lib.save_artifact(artifact, args.artifact_path)
    return artifact


def main():
    args = parse_args()

    run_dir = args.run_dir or os.path.join("runs", args.id_scheme)
    os.makedirs(run_dir, exist_ok=True)

    artifact = maybe_build_artifact(args)

    # vocabs
    query_vocab, prefix_vocab, suffix_vocab = data_lib.build_vocabs_from_artifact(artifact, args.id_scheme)

    train_set = data_lib.GenerativeRetrievalDataset(
        queries=artifact["queries"]["train"],
        docs=artifact["docs"],
        query_vocab=query_vocab,
        prefix_vocab=prefix_vocab,
        suffix_vocab=suffix_vocab,
        id_scheme=args.id_scheme,
    )

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: data_lib.collate_fn(b, pad_id=query_vocab.pad_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        query_vocab_size=len(query_vocab),
        n_prefix=len(prefix_vocab),
        n_suffix=len(suffix_vocab),
        d_model=args.d_model,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    global_step = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {ep}/{args.epochs}")
        total_loss = 0.0

        for batch in pbar:
            q = batch["query_ids"].to(device)
            lengths = batch["lengths"].to(device)
            y_prefix = batch["prefix_id"].to(device)
            y_suffix = batch["suffix_id"].to(device)

            prefix_logits, suffix_logits = model(q, lengths, prefix_ids=y_prefix)
            loss_prefix = F.cross_entropy(prefix_logits, y_prefix)
            loss_suffix = F.cross_entropy(suffix_logits, y_suffix)
            loss = loss_prefix + loss_suffix

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += float(loss.item())
            global_step += 1
            pbar.set_postfix({"loss": f"{total_loss / global_step:.4f}"})

        # epoch end: save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "vocabs": {
                "query": query_vocab.to_dict(),
                "prefix": prefix_vocab.to_dict(),
                "suffix": suffix_vocab.to_dict(),
            },
            "artifact_path": args.artifact_path,
        }
        torch.save(ckpt, os.path.join(run_dir, "ckpt.pt"))

    dt = time.time() - t0
    print(f"Training done. steps={global_step} time={dt:.1f}s  ckpt={os.path.join(run_dir, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
