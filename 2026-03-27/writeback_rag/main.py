from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.pipeline import WritebackRAGPipeline


@dataclass
class QAExample:
    id: str
    question: str
    answer: str


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    root = Path(__file__).resolve().parent
    corpus_path = root / "toy_data" / "corpus.jsonl"
    qa_path = root / "toy_data" / "qa.jsonl"

    corpus = list(load_jsonl(corpus_path))
    qa = [QAExample(**x) for x in load_jsonl(qa_path)]

    pipeline = WritebackRAGPipeline(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=4,
    )

    pipeline.build_index(corpus)

    baseline = pipeline.evaluate(qa)
    print("Baseline accuracy:", f"{baseline.correct}/{baseline.total}", f"({baseline.accuracy:.1%})")

    pipeline.write_back_from_labeled_examples(qa)

    after = pipeline.evaluate(qa)
    print("Writeback accuracy:", f"{after.correct}/{after.total}", f"({after.accuracy:.1%})")


if __name__ == "__main__":
    main()
