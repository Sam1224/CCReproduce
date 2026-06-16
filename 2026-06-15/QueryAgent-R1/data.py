"""
QueryAgent-R1 — Toy Dataset

Generates synthetic user search histories and product inventories
for the e-commerce query recommendation task.

Paper: https://arxiv.org/abs/2606.05671
Task: Given user history, recommend queries whose retrieved products
      match downstream purchase intent (maximize Cons@K = consistency).
"""
import json
import os
import random

PRODUCT_CATEGORIES = [
    "laptop", "headphones", "keyboard", "monitor", "smartphone",
    "sneakers", "backpack", "jacket", "sunglasses", "watch",
    "coffee_maker", "blender", "air_fryer", "cookware", "knife_set",
    "yoga_mat", "dumbbells", "protein_powder", "running_shoes", "resistance_band",
    "skincare_serum", "moisturizer", "sunscreen", "face_mask", "eye_cream",
]

QUERY_TEMPLATES = {
    "laptop": ["best laptop for work", "lightweight laptop", "gaming laptop under budget"],
    "headphones": ["wireless headphones", "noise cancelling earbuds", "best headphones 2026"],
    "keyboard": ["mechanical keyboard", "wireless keyboard", "ergonomic keyboard"],
    "sneakers": ["running sneakers", "casual sneakers", "best sneakers 2026"],
    "coffee_maker": ["best coffee maker", "espresso machine home", "drip coffee maker"],
    "protein_powder": ["best protein powder", "whey protein", "vegan protein supplement"],
    "skincare_serum": ["vitamin c serum", "anti-aging serum", "best face serum"],
    "yoga_mat": ["thick yoga mat", "non-slip yoga mat", "travel yoga mat"],
}


def generate_product_inventory(n: int = 1000) -> list[dict]:
    """Generate toy product inventory."""
    products = []
    for i in range(n):
        cat = random.choice(PRODUCT_CATEGORIES)
        products.append({
            "product_id": f"p{i:04d}",
            "category": cat,
            "title": f"{cat.replace('_', ' ').title()} Model {i}",
            "price": round(random.uniform(15, 800), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "purchase_rate": round(random.uniform(0.01, 0.25), 3),
        })
    return products


def generate_user_history(user_id: int, inventory: list[dict]) -> dict:
    """Generate a user's search and purchase history."""
    preferred_cats = random.sample(PRODUCT_CATEGORIES, k=random.randint(2, 5))
    search_history = []
    purchased_products = []

    for cat in preferred_cats:
        templates = QUERY_TEMPLATES.get(cat, [f"best {cat.replace('_', ' ')}"])
        for tmpl in random.sample(templates, k=min(2, len(templates))):
            search_history.append({
                "query": tmpl,
                "category": cat,
                "clicked": random.choice([True, True, False]),
            })

    # Simulate purchases
    for item in inventory:
        if item["category"] in preferred_cats and random.random() < 0.05:
            purchased_products.append(item["product_id"])

    return {
        "user_id": user_id,
        "preferred_categories": preferred_cats,
        "search_history": search_history,
        "purchased_products": purchased_products[:10],
    }


def build_dataset(output_dir: str = "data", n_users: int = 500, n_products: int = 1000):
    os.makedirs(output_dir, exist_ok=True)

    inventory = generate_product_inventory(n_products)
    users = [generate_user_history(uid, inventory) for uid in range(n_users)]

    # Build ground truth: for each user, what products do they buy given a query
    ground_truth = {}
    for user in users:
        uid = str(user["user_id"])
        ground_truth[uid] = {
            "preferred_categories": user["preferred_categories"],
            "purchased_products": user["purchased_products"],
        }

    with open(os.path.join(output_dir, "inventory.json"), "w") as f:
        json.dump(inventory, f, indent=2)
    with open(os.path.join(output_dir, "users.json"), "w") as f:
        json.dump(users, f, indent=2)
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Dataset: {n_products} products, {n_users} users.")
    return inventory, users, ground_truth


if __name__ == "__main__":
    build_dataset()
