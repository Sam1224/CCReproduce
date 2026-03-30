from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def choose_label(has_nav: bool, has_search: bool, has_footer: bool) -> int:
    # Deterministic “oracle” mapping for synthetic supervision.
    # 0: nav+search, 1: nav-only, 2: search-only, 3: neither.
    if has_nav and has_search:
        return 0
    if has_nav and (not has_search):
        return 1
    if (not has_nav) and has_search:
        return 2
    return 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            has_nav = random.random() < 0.7
            has_search = random.random() < 0.6
            has_footer = random.random() < 0.8
            num_cards = random.randint(2, 10)

            spec = {
                "has_nav": has_nav,
                "has_search": has_search,
                "has_footer": has_footer,
                "num_cards": num_cards,
            }
            label = choose_label(has_nav, has_search, has_footer)
            f.write(json.dumps({"id": f"t{i}", "spec": spec, "label": label}, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
