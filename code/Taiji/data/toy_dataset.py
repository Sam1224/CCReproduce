"""
Toy dataset generator for Taiji reproduction.
Generates synthetic user interaction sequences with semantic labels.
"""
import json
import random
import numpy as np
from pathlib import Path


def generate_item_pool(n_items: int = 1000, n_categories: int = 20) -> list[dict]:
    """Generate a synthetic item (ad creative) pool with semantic attributes."""
    categories = [f"cat_{i}" for i in range(n_categories)]
    items = []
    for item_id in range(n_items):
        items.append({
            "item_id": item_id,
            "category": random.choice(categories),
            "price_tier": random.choice(["low", "mid", "high"]),
            "style_tags": random.sample(["fashion", "tech", "food", "sport", "beauty", "home"], 2),
            "ctr_score": np.random.beta(2, 5),  # historical CTR
            "cvr_score": np.random.beta(1, 10),  # historical CVR
        })
    return items


def generate_user_behavior_sequence(
    user_id: int,
    item_pool: list[dict],
    seq_len: int = 20
) -> dict:
    """
    Generate a user's historical interaction sequence.
    Each interaction has: item, action (click/purchase), timestamp.
    """
    # Each user has a latent preference vector (2 categories they prefer)
    preferred_cats = random.sample([item["category"] for item in item_pool[:20]], 2)
    preferred_price = random.choice(["low", "mid", "high"])

    history = []
    for step in range(seq_len):
        # With probability 0.6, pick from preferred categories
        if random.random() < 0.6:
            candidates = [i for i in item_pool if i["category"] in preferred_cats]
        else:
            candidates = item_pool
        item = random.choice(candidates)

        # Click probability influenced by user preference alignment
        cat_match = item["category"] in preferred_cats
        price_match = item["price_tier"] == preferred_price
        click_prob = 0.3 + 0.3 * cat_match + 0.1 * price_match
        purchase_prob = click_prob * 0.15

        history.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "price_tier": item["price_tier"],
            "style_tags": item["style_tags"],
            "clicked": int(random.random() < click_prob),
            "purchased": int(random.random() < purchase_prob),
            "step": step,
        })

    return {
        "user_id": user_id,
        "preferred_categories": preferred_cats,
        "preferred_price": preferred_price,
        "history": history,
    }


def generate_cot_label(user: dict) -> str:
    """
    Generate reverse-engineered Chain-of-Thought label for a user sequence.
    In Taiji, this is done by an LLM offline; here we use rule-based templates.
    Formula: CoT = reasoning about WHY the user behaves this way.
    """
    preferred_cats = user["preferred_categories"]
    preferred_price = user["preferred_price"]
    purchased_items = [h for h in user["history"] if h["purchased"]]

    cot = (
        f"Analyzing user {user['user_id']}'s behavior sequence:\n"
        f"- The user shows strong interest in categories: {', '.join(preferred_cats)}.\n"
        f"- Price sensitivity: the user tends toward {preferred_price}-tier items.\n"
        f"- Purchase pattern: {len(purchased_items)} purchases out of {len(user['history'])} interactions "
        f"({100*len(purchased_items)//len(user['history'])}% conversion).\n"
        f"- Inferred purchase intent: high-affinity items in {preferred_cats[0]} with {preferred_price} price tier.\n"
        f"Recommendation reasoning: Focus on {preferred_cats[0]} category ads at {preferred_price} price "
        f"to maximize both CTR and CVR alignment."
    )
    return cot


def build_dataset(n_users: int = 500, n_items: int = 500, output_dir: str = "data/toy_data"):
    """Build and save the toy dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    item_pool = generate_item_pool(n_items)

    dataset = []
    for user_id in range(n_users):
        user = generate_user_behavior_sequence(user_id, item_pool)
        cot_label = generate_cot_label(user)
        user["cot_label"] = cot_label
        dataset.append(user)

    # Split train/val/test
    random.shuffle(dataset)
    n_train = int(0.8 * len(dataset))
    n_val = int(0.1 * len(dataset))

    splits = {
        "train": dataset[:n_train],
        "val": dataset[n_train:n_train + n_val],
        "test": dataset[n_train + n_val:],
    }

    for split_name, split_data in splits.items():
        out_path = Path(output_dir) / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} {split_name} samples to {out_path}")

    with open(Path(output_dir) / "items.json", "w") as f:
        json.dump(item_pool, f)
    print(f"Saved {len(item_pool)} items to {output_dir}/items.json")


if __name__ == "__main__":
    build_dataset()
