"""
Synthetic data generator for OCL policy training.

Simulates adversarial buyer-seller negotiation actions.
Each action is a dict with features; label = 1 if the action violates policy.

Action types (as in an e-commerce buyer-seller platform):
  - price_update: change listed price
  - discount_apply: apply coupon/discount
  - contract_modify: modify payment terms
  - refund_request: initiate refund
  - message_send: send negotiation message
  - order_confirm: confirm an order

Violations:
  - price_update with delta > 50% (price manipulation)
  - contract_modify with unauthorized fields
  - discount_apply above authorized ceiling
  - order_confirm while price is disputed
"""

import json
import random
import argparse
import os

ACTION_TYPES = ["price_update", "discount_apply", "contract_modify",
                "refund_request", "message_send", "order_confirm"]

AUTHORIZED_DISCOUNT_MAX = 0.30   # 30% max discount
PRICE_DELTA_MAX = 0.50           # 50% max price change
AUTHORIZED_CONTRACT_FIELDS = {"payment_days", "currency", "installments"}


def _random_action():
    atype = random.choice(ACTION_TYPES)
    action = {"type": atype}

    if atype == "price_update":
        action["delta_pct"] = random.uniform(-0.80, 0.80)  # can be adversarial
    elif atype == "discount_apply":
        action["discount_pct"] = random.uniform(0.0, 0.60)
    elif atype == "contract_modify":
        all_fields = list(AUTHORIZED_CONTRACT_FIELDS) + ["ownership", "liability", "arbitration"]
        n = random.randint(1, 3)
        action["fields"] = random.sample(all_fields, n)
    elif atype == "refund_request":
        action["amount_pct"] = random.uniform(0.0, 1.0)
    elif atype == "message_send":
        action["contains_threat"] = random.random() < 0.1
    elif atype == "order_confirm":
        action["price_disputed"] = random.random() < 0.3

    return action


def is_violation(action: dict) -> int:
    atype = action["type"]
    if atype == "price_update":
        return int(abs(action.get("delta_pct", 0)) > PRICE_DELTA_MAX)
    elif atype == "discount_apply":
        return int(action.get("discount_pct", 0) > AUTHORIZED_DISCOUNT_MAX)
    elif atype == "contract_modify":
        unauthorized = set(action.get("fields", [])) - AUTHORIZED_CONTRACT_FIELDS
        return int(len(unauthorized) > 0)
    elif atype == "message_send":
        return int(action.get("contains_threat", False))
    elif atype == "order_confirm":
        return int(action.get("price_disputed", False))
    return 0


def action_to_feature_vector(action: dict):
    """Convert action dict to a fixed-size float feature vector."""
    atype_onehot = [0.0] * len(ACTION_TYPES)
    atype_onehot[ACTION_TYPES.index(action["type"])] = 1.0

    scalar_features = [
        action.get("delta_pct", 0.0),
        action.get("discount_pct", 0.0),
        float(len(action.get("fields", []))),
        action.get("amount_pct", 0.0),
        float(action.get("contains_threat", False)),
        float(action.get("price_disputed", False)),
    ]
    return atype_onehot + scalar_features  # dim = 6 + 6 = 12


def generate(n_samples: int, out_path: str, seed: int = 42):
    random.seed(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    records = []
    for _ in range(n_samples):
        action = _random_action()
        label = is_violation(action)
        feat = action_to_feature_vector(action)
        records.append({"action": action, "features": feat, "label": label})
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    violations = sum(r["label"] for r in records)
    print(f"Generated {n_samples} samples → {out_path}  "
          f"(violations: {violations} / {n_samples} = {violations/n_samples:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out", default="data/actions.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.n_samples, args.out, args.seed)
