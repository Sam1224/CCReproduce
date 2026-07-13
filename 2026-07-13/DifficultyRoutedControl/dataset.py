import re
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


ACTIONS = ["refund", "cancel", "exchange", "update_address"]
ITEMS = [
    ("black jacket", "outerwear"),
    ("blue sneakers", "footwear"),
    ("green backpack", "bag"),
    ("silver watch", "accessory"),
    ("white dress", "apparel"),
    ("red headset", "electronics"),
]
CITIES = ["Seattle", "Austin", "Boston", "Denver", "Miami", "Dallas"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


@dataclass
class ServiceSession:
    conversation: str
    slot_texts: list[str]
    difficulty: int
    action_id: int
    gold_slot: int


class ToyServiceSessionDataset(Dataset):
    """Synthetic customer-service sessions for difficulty routing and write verification."""

    def __init__(self, samples: int = 512, split: str = "train", vocab: dict[str, int] | None = None):
        self.samples = samples
        self.split = split
        self.seed_offset = 0 if split == "train" else 20_000
        self.sessions = [self._generate_session(index) for index in range(samples)]
        self.vocab = vocab or self._build_vocab(self.sessions)
        self.vector_size = len(self.vocab)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int) -> dict:
        session = self.sessions[index]
        slot_vectors = [self.vectorize(slot_text) for slot_text in session.slot_texts]
        slot_mask = [1.0] * len(slot_vectors)
        while len(slot_vectors) < 2:
            slot_vectors.append(torch.zeros(self.vector_size))
            slot_mask.append(0.0)
        return {
            "conversation_vec": self.vectorize(session.conversation),
            "slot_vectors": torch.stack(slot_vectors[:2]),
            "slot_mask": torch.tensor(slot_mask[:2], dtype=torch.float32),
            "difficulty": torch.tensor(session.difficulty, dtype=torch.long),
            "action_id": torch.tensor(session.action_id, dtype=torch.long),
            "gold_slot": torch.tensor(session.gold_slot, dtype=torch.long),
        }

    def vectorize(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.vector_size)
        for token in tokenize(text):
            token_id = self.vocab.get(token)
            if token_id is not None:
                vector[token_id] += 1.0
        if vector.sum() > 0:
            vector = vector / vector.sum()
        return vector

    def _build_vocab(self, sessions: list[ServiceSession]) -> dict[str, int]:
        vocab_tokens = sorted({token for session in sessions for token in tokenize(session.conversation)} | {token for session in sessions for slot in session.slot_texts for token in tokenize(slot)})
        return {token: idx for idx, token in enumerate(vocab_tokens)}

    def _generate_session(self, index: int) -> ServiceSession:
        generator = random.Random(self.seed_offset + index)
        action_id = index % len(ACTIONS)
        if index % 5 in {0, 1, 2}:
            return self._simple_session(generator, action_id)
        return self._complex_session(generator, action_id)

    def _make_order(self, generator: random.Random, order_offset: int) -> dict:
        item_name, category = ITEMS[(generator.randint(0, 10_000) + order_offset) % len(ITEMS)]
        order_id = f"ORD{generator.randint(1000, 9999)}"
        city = CITIES[(generator.randint(0, 10_000) + order_offset) % len(CITIES)]
        return {
            "order_id": order_id,
            "item_name": item_name,
            "category": category,
            "city": city,
        }

    def _simple_session(self, generator: random.Random, action_id: int) -> ServiceSession:
        order = self._make_order(generator, 0)
        action = ACTIONS[action_id]
        conversation = {
            "refund": f"Hi support, please refund order {order['order_id']} for the {order['item_name']}. The package never arrived and I only need the refund for this order.",
            "cancel": f"Please cancel order {order['order_id']} for the {order['item_name']}. It has not shipped yet, and I do not want anything changed on other orders.",
            "exchange": f"I need to exchange order {order['order_id']} for the {order['item_name']} because the size is wrong. Keep the request attached to this order only.",
            "update_address": f"Update the address for order {order['order_id']} for the {order['item_name']} to {order['city']}. No other orders should be modified.",
        }[action]
        slot_text = f"{order['order_id']} {order['item_name']} {order['category']} primary_request"
        return ServiceSession(conversation=conversation, slot_texts=[slot_text], difficulty=0, action_id=action_id, gold_slot=0)

    def _complex_session(self, generator: random.Random, action_id: int) -> ServiceSession:
        first = self._make_order(generator, 1)
        second = self._make_order(generator, 2)
        action = ACTIONS[action_id]
        if action == "refund":
            conversation = (
                f"I have two related orders: {first['order_id']} for the {first['item_name']} and replacement {second['order_id']} for the {second['item_name']}. "
                f"Please refund the earlier original order {first['order_id']}, not the replacement {second['order_id']}, because the replacement already arrived."
            )
            gold_slot = 0
        elif action == "cancel":
            conversation = (
                f"Keep order {first['order_id']} active, but cancel order {second['order_id']} for the {second['item_name']} if inventory still shows pending. "
                f"The first order {first['order_id']} is my fallback plan and must stay untouched."
            )
            gold_slot = 1
        elif action == "exchange":
            conversation = (
                f"I first asked about exchanging {second['order_id']} for the {second['item_name']}, but after checking the confirmation email I need the exchange on {first['order_id']} for the {first['item_name']}. "
                f"Do not apply the exchange to the later order {second['order_id']}."
            )
            gold_slot = 0
        else:
            conversation = (
                f"There are two addresses on file. Update order {second['order_id']} for the {second['item_name']} to {second['city']}, "
                f"but keep order {first['order_id']} for the {first['item_name']} at its current destination because that one is already packed."
            )
            gold_slot = 1
        slot_texts = [
            f"{first['order_id']} {first['item_name']} {first['category']} original primary",
            f"{second['order_id']} {second['item_name']} {second['category']} replacement later",
        ]
        return ServiceSession(conversation=conversation, slot_texts=slot_texts, difficulty=1, action_id=action_id, gold_slot=gold_slot)
