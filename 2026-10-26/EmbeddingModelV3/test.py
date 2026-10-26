from __future__ import annotations

import torch

from data import all_texts, build_raw_records
from model import SimpleTokenizer, ToyCascadeAnnotator, TwoTowerRetriever, ndcg_at_k


def main() -> None:
    raw_records = build_raw_records()
    annotator = ToyCascadeAnnotator()
    records = annotator.annotate_records(raw_records)

    tokenizer = SimpleTokenizer()
    tokenizer.fit(all_texts(records))
    model = TwoTowerRetriever(tokenizer.vocab_size, hidden_size=64)

    checkpoint = torch.load("checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    first_query = records[0].query
    candidates = [record for record in records if record.query == first_query]
    with torch.no_grad():
        q_batch = tokenizer.batch_encode([first_query] * len(candidates))
        i_batch = tokenizer.batch_encode([record.item_title for record in candidates])
        scores = model.score(q_batch, i_batch).tolist()

    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    grades = [record.teacher_grade for record in candidates]
    ndcg = ndcg_at_k([score for _, score in ranked], grades, k=3)

    print("top-3 for query:", first_query)
    for record, score in ranked[:3]:
        print(f"  {record.item_id:>2} | score={score:.4f} | grade={record.teacher_grade} | {record.item_title}")
    print("ndcg@3:", round(ndcg, 4))


if __name__ == "__main__":
    main()
