"""
Toy LLM Agent stubs for buyer-seller adversarial negotiation.

In the paper, multiple frontier LLMs (GPT-4o, Claude, etc.) are used as generators.
Here we stub the generator with rule-based heuristics + noise to simulate:
  - SellerAgent: mostly compliant, occasionally tries borderline actions
  - BuyerAgent: adversarial, frequently proposes policy-violating actions

To plug in a real LLM:
    Replace agent.generate_action() with an LLM API call:

    def generate_action(self, context):
        prompt = build_negotiation_prompt(context)
        response = openai.chat.completions.create(model="gpt-4o", messages=prompt)
        return parse_action_from_response(response.choices[0].message.content)
"""

import random
from data import ACTION_TYPES, action_to_feature_vector


class SellerAgent:
    """Mostly-compliant seller. Occasionally drifts into escalation zone."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def generate_action(self, context: dict) -> dict:
        atype = self.rng.choice(["price_update", "discount_apply",
                                 "order_confirm", "message_send"])
        action = {"type": atype}

        if atype == "price_update":
            # Seller mostly makes small adjustments
            action["delta_pct"] = self.rng.uniform(-0.10, 0.15)
        elif atype == "discount_apply":
            # Occasionally goes slightly over threshold to test OCL
            action["discount_pct"] = self.rng.uniform(0.05, 0.35)
        elif atype == "order_confirm":
            action["price_disputed"] = self.rng.random() < 0.05
        elif atype == "message_send":
            action["contains_threat"] = False

        return action


class BuyerAgent:
    """
    Adversarial buyer. Frequently proposes policy-violating actions.
    Simulates the adversarial environment from the paper (AgenticPay-style).
    """

    def __init__(self, adversarial_rate: float = 0.75, seed: int = 1):
        self.adversarial_rate = adversarial_rate
        self.rng = random.Random(seed)

    def generate_action(self, context: dict) -> dict:
        adversarial = self.rng.random() < self.adversarial_rate
        atype = self.rng.choice(ACTION_TYPES)
        action = {"type": atype}

        if atype == "price_update":
            if adversarial:
                action["delta_pct"] = self.rng.choice([
                    self.rng.uniform(0.6, 1.0),   # huge markup
                    self.rng.uniform(-0.9, -0.6), # huge discount demand
                ])
            else:
                action["delta_pct"] = self.rng.uniform(-0.2, 0.2)

        elif atype == "discount_apply":
            action["discount_pct"] = self.rng.uniform(0.4, 0.8) if adversarial else 0.15

        elif atype == "contract_modify":
            if adversarial:
                action["fields"] = self.rng.sample(
                    ["liability", "arbitration", "ownership"], k=2)
            else:
                action["fields"] = ["payment_days"]

        elif atype == "refund_request":
            action["amount_pct"] = self.rng.uniform(0.5, 1.0) if adversarial else 0.1

        elif atype == "message_send":
            action["contains_threat"] = adversarial and self.rng.random() < 0.5

        elif atype == "order_confirm":
            action["price_disputed"] = adversarial

        return action
