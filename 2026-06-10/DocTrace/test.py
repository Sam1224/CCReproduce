import argparse
import json
import os

import numpy as np
import torch

from agents import (
    AggregatorAgent,
    AnswerAgent,
    FilterAgent,
    FilterModel,
    RetrieverAgent,
    token_cost,
)


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def baseline_topk(paragraphs: list[str], query: str, top_k: int):
    r = RetrieverAgent()
    cands = r.retrieve(paragraphs, query, top_k=top_k)
    ctx = "\n".join([c.text for c in cands])
    return ctx, {"retrieved": [c.__dict__ for c in cands]}


def doctrace_pipeline(paragraphs: list[str], query: str, filter_agent: FilterAgent, retr_top_k: int, filt_top_k: int):
    retr = RetrieverAgent()
    agg = AggregatorAgent()

    retrieved = retr.retrieve(paragraphs, query, top_k=retr_top_k)
    filtered = filter_agent.filter(retrieved, query, top_k=filt_top_k)
    ctx = agg.aggregate(filtered)

    trace = {
        "retrieved": [c.__dict__ for c in retrieved],
        "filtered": [c.__dict__ for c in filtered],
        "aggregated_token_cost": token_cost(ctx),
    }

    return ctx, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--baseline_top_k", type=int, default=8)
    parser.add_argument("--retr_top_k", type=int, default=8)
    parser.add_argument("--filt_top_k", type=int, default=3)
    parser.add_argument("--out_trace", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join(args.ckpt_dir, "ckpt.pt"), map_location="cpu")
    model = FilterModel(dim=int(ckpt["dim"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    filter_agent = FilterAgent(model=model, dim=int(ckpt["dim"]), device=device)
    answer_agent = AnswerAgent()

    test_path = os.path.join(args.data_dir, "test.jsonl")
    records = list(load_jsonl(test_path))

    base_correct = 0
    doc_correct = 0
    base_cost = 0
    doc_cost = 0

    traces = []

    for r in records:
        query = r["query"]
        gold = r["answer"]
        paragraphs = r["paragraphs"]

        base_ctx, base_trace = baseline_topk(paragraphs, query, top_k=args.baseline_top_k)
        base_ans = answer_agent.answer(query, base_ctx)

        doc_ctx, doc_trace = doctrace_pipeline(
            paragraphs,
            query,
            filter_agent=filter_agent,
            retr_top_k=args.retr_top_k,
            filt_top_k=args.filt_top_k,
        )
        doc_ans = answer_agent.answer(query, doc_ctx)

        base_correct += int(base_ans == gold)
        doc_correct += int(doc_ans == gold)

        base_cost += token_cost(base_ctx)
        doc_cost += token_cost(doc_ctx)

        traces.append(
            {
                "doc_id": r["doc_id"],
                "query": query,
                "gold": gold,
                "baseline": {
                    "answer": base_ans,
                    "token_cost": token_cost(base_ctx),
                    "trace": base_trace,
                },
                "doctrace": {
                    "answer": doc_ans,
                    "token_cost": token_cost(doc_ctx),
                    "trace": doc_trace,
                },
            }
        )

    n = len(records)
    print(
        json.dumps(
            {
                "baseline": {
                    "accuracy": base_correct / n,
                    "avg_token_cost": base_cost / n,
                    "top_k": args.baseline_top_k,
                },
                "doctrace_style": {
                    "accuracy": doc_correct / n,
                    "avg_token_cost": doc_cost / n,
                    "retr_top_k": args.retr_top_k,
                    "filt_top_k": args.filt_top_k,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.out_trace:
        os.makedirs(os.path.dirname(args.out_trace), exist_ok=True)
        with open(args.out_trace, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        print(f"trace saved: {args.out_trace}")


if __name__ == "__main__":
    main()
