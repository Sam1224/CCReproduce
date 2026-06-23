"""
Reverse-Engineered CoT Data Generation for Taiji Stage 1 (SFT).

Key idea: Instead of asking LLM to predict next items (which causes ID memorization),
we reverse-engineer: given a user's behavior sequence (including purchases), ask LLM
to reason WHY the user made those choices → generates semantic reasoning chains.

This is the "inverse problem": behavior sequence → intent reasoning, rather than
intent → next behavior. This lets LLM learn causal semantic patterns, not ID patterns.
"""
import json
from pathlib import Path
from typing import Optional


def build_reverse_cot_prompt(user: dict) -> str:
    """
    Build a reverse-engineering prompt for CoT generation.

    The "reverse" direction: given observed purchases, explain the user's reasoning.
    This is fundamentally different from standard recommendation prompting.
    """
    # Only show items the user actually interacted with
    clicked = [h for h in user["history"] if h["clicked"]]
    purchased = [h for h in user["history"] if h["purchased"]]

    history_lines = []
    for h in clicked[-10:]:  # last 10 clicked items
        status = "✓ purchased" if h["purchased"] else "viewed"
        history_lines.append(
            f"  [{status}] category={h['category']}, price={h['price_tier']}, "
            f"style={','.join(h['style_tags'])}"
        )

    purchase_summary = ""
    if purchased:
        cats = [p["category"] for p in purchased]
        prices = [p["price_tier"] for p in purchased]
        purchase_summary = (
            f"The user ultimately purchased items in categories: {set(cats)}, "
            f"at price tiers: {set(prices)}."
        )

    prompt = (
        "You are an expert analyst studying user shopping behavior.\n"
        "Given the following user interaction history and purchase outcomes, "
        "reason step-by-step about this user's purchase intent and preferences.\n\n"
        f"User interaction history:\n" + "\n".join(history_lines) + "\n\n"
        f"{purchase_summary}\n\n"
        "Provide a detailed reasoning chain about:\n"
        "1. What categories and styles does this user genuinely prefer?\n"
        "2. What is their price sensitivity?\n"
        "3. What types of ads/recommendations would best match their purchase intent?\n"
        "Reasoning:"
    )
    return prompt


def build_sft_training_sample(user: dict, cot_label: str) -> dict:
    """
    Build a training sample for SFT: (prompt, completion) pair.
    The completion is the reverse-engineered CoT.
    """
    return {
        "prompt": build_reverse_cot_prompt(user),
        "completion": cot_label,
        "user_id": user["user_id"],
    }


def generate_sft_dataset(
    user_data_path: str,
    output_path: str,
    use_rule_based_cot: bool = True,
) -> None:
    """
    Generate SFT dataset from user interaction data.

    In production Taiji:
    - A large LLM (e.g. GPT-4, Qwen-72B) generates CoT labels offline
    - Rejection sampling filters low-quality CoTs
    Here: use rule-based CoT generation as a proxy.
    """
    with open(user_data_path) as f:
        users = json.load(f)

    sft_samples = []
    for user in users:
        if use_rule_based_cot:
            # Use pre-generated CoT from dataset
            cot = user.get("cot_label", "")
        else:
            # Placeholder: in production, call LLM here
            cot = f"[LLM-generated CoT for user {user['user_id']}]"

        if cot:
            sample = build_sft_training_sample(user, cot)
            sft_samples.append(sample)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sft_samples, f, indent=2)
    print(f"Generated {len(sft_samples)} SFT samples → {output_path}")


if __name__ == "__main__":
    generate_sft_dataset(
        user_data_path="data/toy_data/train.json",
        output_path="data/toy_data/sft_train.json",
    )
    generate_sft_dataset(
        user_data_path="data/toy_data/val.json",
        output_path="data/toy_data/sft_val.json",
    )
