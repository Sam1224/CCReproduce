from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import RetrievalTripletDataset, ensure_toy_data, load_eval_records, load_train_rows
from model import BiEncoderConfig, DenseRetriever
from test import evaluate


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="toy_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-per-channel", type=int, default=20)
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--curriculum", choices=["on", "off"], default="on")
    parser.add_argument("--stage-max-difficulty", default="3,4,5")
    parser.add_argument("--stage-epochs", default="3,2,2")
    parser.add_argument("--stage-lr", default="2e-3,1e-3,5e-4")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--checkpoint-out", default="checkpoints/scaling_dense_retrieval.pt")
    return parser.parse_args()


def build_stage_plan(args: argparse.Namespace) -> list[tuple[int, int, float]]:
    if args.curriculum == "off":
        return [(5, sum(parse_csv_ints(args.stage_epochs)), parse_csv_floats(args.stage_lr)[0])]
    max_difficulties = parse_csv_ints(args.stage_max_difficulty)
    epochs = parse_csv_ints(args.stage_epochs)
    learning_rates = parse_csv_floats(args.stage_lr)
    if not (len(max_difficulties) == len(epochs) == len(learning_rates)):
        raise ValueError("stage-max-difficulty, stage-epochs, and stage-lr must have the same length")
    return list(zip(max_difficulties, epochs, learning_rates))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dir = ensure_toy_data(
        args.data_dir,
        seed=args.seed,
        topk_per_channel=args.topk_per_channel,
        force=args.generate_data,
    )
    train_rows = load_train_rows(data_dir)
    dev_records = load_eval_records(data_dir, "dev")

    cfg = BiEncoderConfig(vocab_size=args.vocab_size, d_model=args.dim, max_len=args.max_len)
    model = DenseRetriever(cfg)
    stage_plan = build_stage_plan(args)

    best_ndcg = -1.0
    best_state = None
    history: list[dict[str, float | int]] = []

    for stage_index, (max_difficulty, epochs, learning_rate) in enumerate(stage_plan, start=1):
        dataset = RetrievalTripletDataset(train_rows, max_difficulty=max_difficulty, max_len=cfg.max_len, vocab_size=cfg.vocab_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in loader:
                pos_scores = model.score_pairs(batch["q_ids"], batch["pos_ids"])
                neg_scores = model.score_pairs(batch["q_ids"], batch["neg_ids"])
                base_loss = F.relu(args.margin - pos_scores + neg_scores)
                sample_weight = 1.0 + 0.15 * (batch["pos_grade"] - 3.0).clamp(min=0.0) + 0.08 * (batch["difficulty"].float() - 1.0)
                loss = (base_loss * sample_weight).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            metrics = evaluate(model, dev_records, k=10, vocab_size=cfg.vocab_size, max_len=cfg.max_len)
            history.append(
                {
                    "stage": stage_index,
                    "max_difficulty": max_difficulty,
                    "epoch": epoch,
                    "loss": epoch_loss / max(len(loader), 1),
                    "ndcg@10": metrics["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "embarrassing@1": metrics["embarrassing@1"],
                }
            )
            print(
                f"stage={stage_index} max_diff={max_difficulty} epoch={epoch} "
                f"loss={epoch_loss / max(len(loader), 1):.4f} ndcg@10={metrics['ndcg@10']:.4f} "
                f"recall@10={metrics['recall@10']:.4f} embarrassing@1={metrics['embarrassing@1']:.4f}"
            )
            if metrics["ndcg@10"] > best_ndcg:
                best_ndcg = metrics["ndcg@10"]
                best_state = {
                    "state": model.state_dict(),
                    "meta": {
                        "vocab_size": cfg.vocab_size,
                        "d_model": cfg.d_model,
                        "max_len": cfg.max_len,
                    },
                    "history": history,
                }

    output_path = Path(args.checkpoint_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    torch.save(best_state, output_path)
    print(json.dumps({"checkpoint": str(output_path), "best_ndcg@10": best_ndcg, "history": history}, indent=2))


if __name__ == "__main__":
    main()
