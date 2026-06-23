"""
Toy cross-domain dataset generator for AIR.

Simulates the content (short video / live stream) → e-commerce purchase scenario:
- Content domain: user watches short videos / follows creators
- E-commerce domain: user buys products that appear in or relate to the content

The key challenge AIR solves: most content interactions are entertainment-driven,
not purchase-intent-driven. AIUs filter out noise.
"""
import json
import random
import numpy as np
from pathlib import Path


CONTENT_CATEGORIES = [
    "fashion_vlog", "cooking_tutorial", "tech_review", "fitness", "beauty_tutorial",
    "home_decor", "gaming", "travel", "book_review", "pet_care",
]

PRODUCT_CATEGORIES = [
    "clothing", "food_ingredient", "electronics", "gym_equipment", "cosmetics",
    "furniture", "game_accessory", "travel_gear", "book", "pet_supply",
]

# Cross-domain affinity: content cat → likely product cat
CONTENT_TO_PRODUCT = {
    "fashion_vlog": ["clothing", "cosmetics"],
    "cooking_tutorial": ["food_ingredient", "furniture"],
    "tech_review": ["electronics", "game_accessory"],
    "fitness": ["gym_equipment", "clothing"],
    "beauty_tutorial": ["cosmetics", "clothing"],
    "home_decor": ["furniture", "home_decor"],
    "gaming": ["game_accessory", "electronics"],
    "travel": ["travel_gear", "clothing"],
    "book_review": ["book"],
    "pet_care": ["pet_supply"],
}


def generate_content_item(item_id: int) -> dict:
    content_cat = random.choice(CONTENT_CATEGORIES)
    return {
        "content_id": item_id,
        "category": content_cat,
        "creator_tier": random.choice(["nano", "micro", "macro", "mega"]),
        "engagement_rate": float(np.random.beta(2, 8)),
        "topic_tags": random.sample(content_cat.split("_") + ["lifestyle", "trending", "review"], 2),
    }


def generate_product_item(item_id: int) -> dict:
    product_cat = random.choice(PRODUCT_CATEGORIES)
    return {
        "product_id": item_id,
        "category": product_cat,
        "price_tier": random.choice(["budget", "mid", "premium"]),
        "ctr_score": float(np.random.beta(2, 5)),
        "cvr_score": float(np.random.beta(1, 10)),
    }


def generate_user(user_id: int, content_pool: list, product_pool: list, seq_len: int = 25) -> dict:
    """
    Generate a user with:
    - Content interaction history (watches, likes, follows)
    - E-commerce purchase history (the cross-domain signal)
    - A latent "true intent" that bridges the two domains
    """
    # Latent preference: 2 content categories → linked product categories
    preferred_content_cats = random.sample(CONTENT_CATEGORIES, 2)
    preferred_product_cats = []
    for cc in preferred_content_cats:
        preferred_product_cats.extend(CONTENT_TO_PRODUCT.get(cc, []))
    preferred_product_cats = list(set(preferred_product_cats))

    # Content history: mostly noise, some signal
    content_history = []
    for step in range(seq_len):
        # 50% chance of preferred category (signal), 50% noise
        if random.random() < 0.5:
            candidates = [c for c in content_pool if c["category"] in preferred_content_cats]
        else:
            candidates = content_pool
        item = random.choice(candidates)

        is_preferred = item["category"] in preferred_content_cats
        watch_prob = 0.4 + 0.3 * is_preferred
        like_prob = watch_prob * 0.3
        follow_prob = like_prob * 0.2

        content_history.append({
            "content_id": item["content_id"],
            "category": item["category"],
            "creator_tier": item["creator_tier"],
            "topic_tags": item["topic_tags"],
            "watched": int(random.random() < watch_prob),
            "liked": int(random.random() < like_prob),
            "followed_creator": int(random.random() < follow_prob),
            "step": step,
        })

    # E-commerce history: cross-domain purchases
    purchase_history = []
    for _ in range(random.randint(1, 6)):
        if random.random() < 0.7 and preferred_product_cats:
            candidates = [p for p in product_pool if p["category"] in preferred_product_cats]
            if not candidates:
                candidates = product_pool
        else:
            candidates = product_pool
        product = random.choice(candidates)
        purchase_history.append({
            "product_id": product["product_id"],
            "category": product["category"],
            "price_tier": product["price_tier"],
        })

    # Ground-truth AIU (rule-based; in production: LLM-generated)
    aiu_label = generate_aiu_label(content_history, purchase_history, preferred_content_cats, preferred_product_cats)

    return {
        "user_id": user_id,
        "preferred_content_cats": preferred_content_cats,
        "preferred_product_cats": preferred_product_cats,
        "content_history": content_history,
        "purchase_history": purchase_history,
        "aiu_label": aiu_label,
    }


def generate_aiu_label(
    content_history: list,
    purchase_history: list,
    preferred_content: list,
    preferred_products: list,
) -> str:
    """
    Generate Atomic Intent Unit label (rule-based proxy).
    In production AIR: an LLM generates these from compressed behavior.
    AIUs are 1-3 short sentences capturing the user's cross-domain purchase intent.
    """
    # Count interaction signals
    watched = [h for h in content_history if h["watched"]]
    liked = [h for h in content_history if h["liked"]]
    followed = [h for h in content_history if h["followed_creator"]]

    content_cats_watched = list({h["category"] for h in watched})[:2]
    purchased_cats = list({p["category"] for p in purchase_history})[:2]

    # Build compact intent description (Atomic Intent Units style)
    aiu_parts = []
    if preferred_content:
        aiu_parts.append(
            f"User shows strong content engagement with {', '.join(preferred_content[:2])} topics "
            f"({len(liked)} likes, {len(followed)} creator follows)."
        )
    if purchased_cats:
        aiu_parts.append(
            f"Cross-domain purchase intent: {', '.join(purchased_cats)} products "
            f"based on {len(purchase_history)} prior purchases."
        )
    if preferred_products:
        aiu_parts.append(
            f"High-signal intent: recommend {', '.join(preferred_products[:2])} category items."
        )

    return " ".join(aiu_parts) if aiu_parts else "Insufficient behavioral signal for intent extraction."


def build_dataset(
    n_users: int = 600,
    n_content: int = 500,
    n_products: int = 500,
    output_dir: str = "data/toy_data",
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    content_pool = [generate_content_item(i) for i in range(n_content)]
    product_pool = [generate_product_item(i) for i in range(n_products)]

    dataset = [generate_user(uid, content_pool, product_pool) for uid in range(n_users)]
    random.shuffle(dataset)

    n_train = int(0.8 * n_users)
    n_val = int(0.1 * n_users)
    splits = {
        "train": dataset[:n_train],
        "val": dataset[n_train : n_train + n_val],
        "test": dataset[n_train + n_val :],
    }

    for split, data in splits.items():
        out = Path(output_dir) / f"{split}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} {split} users → {out}")

    with open(Path(output_dir) / "content_pool.json", "w") as f:
        json.dump(content_pool, f)
    with open(Path(output_dir) / "product_pool.json", "w") as f:
        json.dump(product_pool, f)
    print(f"Saved {n_content} content items and {n_products} products.")


if __name__ == "__main__":
    build_dataset()
