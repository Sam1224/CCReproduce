from __future__ import annotations

from typing import Dict, List, Tuple

from attack import apply_prompt_injection, poison_corpus
from generator import toy_generate
from retriever import Retriever
from toy_data import build_toy_corpus, build_toy_qa, poison_passage


def eval_setting(name: str, use_poison: bool, use_injection: bool) -> Dict[str, float]:
    corpus = build_toy_corpus()
    qa = build_toy_qa()

    if use_poison:
        corpus = poison_corpus(corpus, poison_passage(target_answer="365 days"))

    r = Retriever(dim=256)
    r.build([(d.doc_id, d.text) for d in corpus])

    correct = 0
    attacked = 0

    for q in qa:
        query = q.question
        if use_injection:
            query = apply_prompt_injection(query)

        retrieved = r.search(query, top_k=3)
        pred = toy_generate(q.question, retrieved)

        if pred.strip().lower() == q.answer.strip().lower():
            correct += 1

        # attack success: we force wrong targeted answer for return question
        if q.qid == "q1" and pred.strip().lower() == "365 days":
            attacked += 1

    return {
        "accuracy": correct / len(qa),
        "attack_success_on_q1": attacked,
    }


def main() -> None:
    settings = [
        ("baseline", False, False),
        ("poison_only", True, False),
        ("injection_only", False, True),
        ("PIDP_combined", True, True),
    ]

    for name, p, inj in settings:
        out = eval_setting(name, p, inj)
        print(name, out)


if __name__ == "__main__":
    main()
