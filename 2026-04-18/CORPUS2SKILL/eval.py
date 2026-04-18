import argparse
import json
from pathlib import Path

from tqdm import tqdm

from corpus2skill.agent import Corpus2SkillAgent
from corpus2skill.metrics import token_f1


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True, help="Path to JSONL QA set")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory produced by train.py")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    agent = Corpus2SkillAgent.load(out_dir)

    total = 0
    hit = 0
    f1_sum = 0.0

    for ex in tqdm(list(iter_jsonl(Path(args.qa))), desc="eval"):
        total += 1
        question = ex["question"]
        gold_doc_id = ex["gold_doc_id"]
        gold_answer = ex["gold_answer"]

        results = agent.search(question, top_k=args.top_k)
        retrieved_ids = [r.doc_id for r in results]
        if gold_doc_id in retrieved_ids:
            hit += 1

        # Non-LLM answer baseline: use the first retrieved doc’s highlighted snippet.
        pred_answer = results[0].snippet if results else ""
        f1_sum += token_f1(pred_answer, gold_answer)

    recall_at_k = hit / max(total, 1)
    avg_f1 = f1_sum / max(total, 1)

    print(
        json.dumps(
            {
                "examples": total,
                f"recall@{args.top_k}": recall_at_k,
                "avg_token_f1": avg_f1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
