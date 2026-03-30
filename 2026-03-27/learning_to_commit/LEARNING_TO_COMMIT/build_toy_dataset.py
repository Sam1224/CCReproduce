from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from patches import PRPlan, score_oracle
from repo_memory import RepoMemory


def random_memory(rng: random.Random) -> RepoMemory:
    return RepoMemory(
        tests_required=rng.random() < 0.75,
        docs_required=rng.random() < 0.4,
        prefer_small_diff=rng.random() < 0.6,
    )


def random_issue_type(rng: random.Random) -> str:
    return rng.choice(["bugfix", "feature", "refactor"])


def sample_candidates(rng: random.Random, k: int = 5) -> list[PRPlan]:
    out = []
    for _ in range(k):
        out.append(
            PRPlan(
                has_tests=rng.random() < 0.6,
                updates_docs=rng.random() < 0.3,
                diff_lines=int(rng.randint(20, 200)),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            mem = random_memory(rng)
            issue_type = random_issue_type(rng)
            cands = sample_candidates(rng, k=args.k)

            scores = [score_oracle(mem, c) for c in cands]
            label = int(max(range(len(scores)), key=lambda j: scores[j]))

            f.write(
                json.dumps(
                    {
                        "id": f"e{i}",
                        "issue_type": issue_type,
                        "memory": {
                            "tests_required": mem.tests_required,
                            "docs_required": mem.docs_required,
                            "prefer_small_diff": mem.prefer_small_diff,
                        },
                        "candidates": [c.__dict__ for c in cands],
                        "label": label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
