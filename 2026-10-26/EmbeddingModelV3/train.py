from __future__ import annotations

from collections import Counter

import torch

from data import all_texts, build_raw_records, curriculum_triplets, group_by_query, mine_curriculum_examples
from model import (
    SimpleTokenizer,
    ToyCascadeAnnotator,
    TwoTowerRetriever,
    bce_stage_loss,
    multiple_negative_ranking_loss,
    ndcg_at_k,
    triplet_stage_loss,
)


def evaluate(model: TwoTowerRetriever, tokenizer: SimpleTokenizer, records) -> float:
    grouped = group_by_query(records)
    ndcgs = []
    model.eval()
    with torch.no_grad():
        for query, query_records in grouped.items():
            q_batch = tokenizer.batch_encode([query] * len(query_records))
            i_batch = tokenizer.batch_encode([record.item_title for record in query_records])
            scores = model.score(q_batch, i_batch).tolist()
            grades = [record.teacher_grade for record in query_records]
            ndcgs.append(ndcg_at_k(scores, grades, k=3))
    return sum(ndcgs) / max(len(ndcgs), 1)


def main() -> None:
    torch.manual_seed(42)

    raw_records = build_raw_records()
    annotator = ToyCascadeAnnotator()
    records = annotator.annotate_records(raw_records)

    buckets = mine_curriculum_examples(records)
    bucket_sizes = {name: len(items) for name, items in buckets.items()}
    print("bucket sizes:", bucket_sizes)
    print("cascade coverage:", Counter(trace[-1] for record in records for trace in [record.cascade_trace]))

    tokenizer = SimpleTokenizer()
    tokenizer.fit(all_texts(records))
    model = TwoTowerRetriever(tokenizer.vocab_size, hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    print("initial ndcg@3:", round(evaluate(model, tokenizer, records), 4))

    for step in range(120):
        optimizer.zero_grad()
        loss = bce_stage_loss(model, tokenizer, buckets["easy_pos"], buckets["random_neg"][: len(buckets["easy_pos"])])
        loss.backward()
        optimizer.step()
        if step in {0, 59, 119}:
            print(f"stage1 step={step+1} loss={loss.item():.4f}")

    triplets = curriculum_triplets(records)
    for step in range(120):
        optimizer.zero_grad()
        loss = multiple_negative_ranking_loss(model, tokenizer, triplets["stage2"])
        loss.backward()
        optimizer.step()
        if step in {0, 59, 119}:
            print(f"stage2 step={step+1} loss={loss.item():.4f}")

    for step in range(120):
        optimizer.zero_grad()
        loss = triplet_stage_loss(model, tokenizer, triplets["stage3"])
        loss.backward()
        optimizer.step()
        if step in {0, 59, 119}:
            print(f"stage3 step={step+1} loss={loss.item():.4f}")

    final_ndcg = evaluate(model, tokenizer, records)
    print("final ndcg@3:", round(final_ndcg, 4))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": tokenizer.stoi,
            "metrics": {"ndcg_at_3": final_ndcg},
        },
        "checkpoint.pt",
    )
    print("saved checkpoint.pt")


if __name__ == "__main__":
    main()
