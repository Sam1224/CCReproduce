import argparse
import json
import random
from pathlib import Path

from dataset import (
    RuleBasedLLMStub,
    build_listing_text,
    load_listings,
    load_seed_queries,
    load_sessions,
)


def generate_query(seed_query: str, pos: dict, neg: dict, llm: RuleBasedLLMStub) -> str:
    """Seed-guided contrastive query generation (deterministic stub).

    The paper uses LLM prompting with: (seed query + contrastive listing pair).
    Here we approximate with attribute-difference-driven templating.
    """

    pos_amenities = set(pos.get("amenities", []))
    neg_amenities = set(neg.get("amenities", []))
    pos_only = sorted(pos_amenities - neg_amenities)

    amenity_hint = pos_only[0] if pos_only else (pos.get("amenities") or [""])[0]
    property_type = pos.get("property_type", "place")
    city = pos.get("city", "")
    vibe = (pos.get("vibe") or [""])[0]

    prompt = {
        "seed_query": seed_query,
        "pos": pos,
        "neg": neg,
        "hints": {
            "amenity": amenity_hint,
            "property_type": property_type,
            "city": city,
            "vibe": vibe,
        },
    }
    return llm.generate(prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listings", default="toy_data/listings.jsonl")
    parser.add_argument("--sessions", default="toy_data/sessions.jsonl")
    parser.add_argument("--seeds", default="toy_data/seed_queries.txt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args()

    listings = load_listings(args.listings)
    sessions = load_sessions(args.sessions)
    seeds = load_seed_queries(args.seeds)
    llm = RuleBasedLLMStub()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in sessions:
            if written >= args.max:
                break
            pos = listings[s["pos"]]
            neg = listings[s["neg"]]
            seed = random.choice(seeds)
            query = generate_query(seed, pos, neg, llm)
            record = {
                "query": query,
                "pos_id": pos["listing_id"],
                "neg_id": neg["listing_id"],
                "pos_text": build_listing_text(pos),
                "neg_text": build_listing_text(neg),
                "labeling": {
                    "contrastive_by_construction": {"pos": 1, "neg": 0},
                    "virtual_judge": {
                        "pos_score": llm.virtual_judge_score(query, pos),
                        "neg_score": llm.virtual_judge_score(query, neg),
                    },
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} triples -> {out_path}")


if __name__ == "__main__":
    main()
