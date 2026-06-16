"""Toy dataset for AIR: cross-domain (content → e-commerce) recommendation."""
import json
import random
import os

CONTENT_CATEGORIES = [
    "cooking", "fitness", "fashion", "travel", "gaming",
    "beauty", "tech_review", "home_decor", "pets", "music",
]

ECOM_CATEGORIES = {
    "cooking": ["kitchen_appliance", "cookbook", "ingredient", "cooking_tool"],
    "fitness": ["sportswear", "supplement", "gym_equipment", "fitness_tracker"],
    "fashion": ["clothing", "accessories", "shoes", "jewelry"],
    "travel": ["luggage", "travel_gadget", "outdoor_gear", "travel_accessory"],
    "gaming": ["game", "gaming_peripheral", "console", "gaming_chair"],
    "beauty": ["skincare", "makeup", "haircare", "fragrance"],
    "tech_review": ["smartphone", "laptop", "smart_home", "wearable"],
    "home_decor": ["furniture", "lighting", "plants", "art"],
    "pets": ["pet_food", "pet_toy", "grooming_tool", "pet_accessory"],
    "music": ["instrument", "headphone", "audio_equipment", "music_software"],
}


def generate_content_behavior(user_id: int, n_events: int = 10) -> list[dict]:
    """Generate synthetic content interaction events for a user."""
    events = []
    preferred_cats = random.sample(CONTENT_CATEGORIES, k=min(3, len(CONTENT_CATEGORIES)))
    for _ in range(n_events):
        cat = random.choice(preferred_cats)
        events.append({
            "user_id": user_id,
            "item_id": f"content_{cat}_{random.randint(1000, 9999)}",
            "category": cat,
            "action": random.choice(["view", "like", "share", "comment"]),
            "duration_seconds": random.randint(5, 300),
            "timestamp": random.randint(1700000000, 1700086400),
        })
    return events


def generate_ecom_items(n_items: int = 500) -> list[dict]:
    """Generate synthetic e-commerce items."""
    items = []
    for i in range(n_items):
        content_cat = random.choice(CONTENT_CATEGORIES)
        ecom_cat = random.choice(ECOM_CATEGORIES[content_cat])
        items.append({
            "item_id": f"ecom_{i:04d}",
            "title": f"{ecom_cat.replace('_', ' ').title()} Product {i}",
            "category": ecom_cat,
            "source_content_category": content_cat,
            "price": round(random.uniform(10, 500), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "description": f"High quality {ecom_cat} for {content_cat} enthusiasts.",
        })
    return items


def generate_ground_truth(users: list[dict], items: list[dict]) -> dict:
    """Generate ground truth purchase labels."""
    gt = {}
    item_map = {it["item_id"]: it for it in items}
    for u in users:
        uid = u["user_id"]
        # Users tend to buy items related to their content interests
        preferred_cats = list({e["category"] for e in u["events"]})
        relevant_items = [
            it["item_id"] for it in items
            if it["source_content_category"] in preferred_cats
        ]
        gt[str(uid)] = random.sample(relevant_items, k=min(5, len(relevant_items)))
    return gt


def save_dataset(output_dir: str = "data", n_users: int = 200, n_items: int = 500):
    os.makedirs(output_dir, exist_ok=True)

    users = []
    for uid in range(n_users):
        events = generate_content_behavior(uid)
        users.append({"user_id": uid, "events": events})

    items = generate_ecom_items(n_items)
    gt = generate_ground_truth(users, items)

    # Save content behaviors
    behaviors = [e for u in users for e in u["events"]]
    with open(os.path.join(output_dir, "content_behaviors.json"), "w") as f:
        json.dump(behaviors, f, indent=2)

    # Save e-commerce items
    with open(os.path.join(output_dir, "ecom_items.json"), "w") as f:
        json.dump(items, f, indent=2)

    # Save ground truth
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    print(f"Saved {len(behaviors)} behaviors, {len(items)} items, {len(gt)} user ground truths.")
    return users, items, gt


if __name__ == "__main__":
    save_dataset()
