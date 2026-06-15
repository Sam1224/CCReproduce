"""
OneRetrieval – Toy reproduction of core ideas from:
  "OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model"
  Kuaishou Technology, arXiv 2606.13533

Key concepts reproduced:
  1. Keyword-Aligned Encoding (KAE) – each identifier position maps to a
     human-readable attribute word rather than an opaque quantisation index.
  2. Reserved Slots – codebook contains extra slots that can be bound to new
     keywords at deployment time without retraining.
  3. Generative retrieval – seq2seq model generates item identifiers
     autoregressively from a query.

Reference formula (KAE identifier construction):
  I(item) = [k1, k2, ..., kL]   where ki ∈ V_attr ∪ V_reserved
  V_reserved = {r1, r2, ..., rR} (pre-allocated, slot_i is unbound by default)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Attribute vocabulary and reserved slots
# ---------------------------------------------------------------------------

BASE_ATTRIBUTES = [
    "fashion", "electronics", "sports", "beauty", "food",
    "shirt", "phone", "sneaker", "lipstick", "organic",
    "men", "women", "kids", "premium", "budget",
    "red", "blue", "black", "white", "size-M",
]

NUM_RESERVED_SLOTS = 20
RESERVED_PREFIX = "reserved_"

class AttributeCodebook:
    """
    Maps attribute words ↔ slot indices.
    Reserved slots can be bound to new keywords post-deployment.
    """

    def __init__(self) -> None:
        self._word2id: dict[str, int] = {}
        self._id2word: dict[int, str] = {}
        # Build base vocabulary
        for i, word in enumerate(BASE_ATTRIBUTES):
            self._word2id[word] = i
            self._id2word[i] = word
        # Add reserved slots
        base_size = len(BASE_ATTRIBUTES)
        for j in range(NUM_RESERVED_SLOTS):
            slot_name = f"{RESERVED_PREFIX}{j}"
            idx = base_size + j
            self._word2id[slot_name] = idx
            self._id2word[idx] = slot_name

    @property
    def size(self) -> int:
        return len(self._word2id)

    def bind_reserved_slot(self, slot_idx: int, new_word: str) -> None:
        """
        Bind a new keyword to a reserved slot at runtime (no retraining needed).
        This is the core 'editable retrieval' mechanism.
        """
        slot_name = f"{RESERVED_PREFIX}{slot_idx}"
        if slot_name not in self._word2id:
            raise ValueError(f"Reserved slot {slot_idx} does not exist.")
        internal_id = self._word2id[slot_name]
        # Re-point the slot to the new word
        del self._word2id[slot_name]
        self._word2id[new_word] = internal_id
        self._id2word[internal_id] = new_word
        print(f"[Codebook] Bound reserved_slot_{slot_idx} → '{new_word}' (no retrain needed)")

    def encode(self, word: str) -> int:
        return self._word2id.get(word, -1)

    def decode(self, idx: int) -> str:
        return self._id2word.get(idx, "<UNK>")


# ---------------------------------------------------------------------------
# 2. Keyword-Aligned Identifier (KAE)
# ---------------------------------------------------------------------------

IDENTIFIER_LENGTH = 4   # L – each position covers a different attribute group

@dataclass
class Item:
    item_id: str
    attributes: list[str]          # human-readable attributes, len == IDENTIFIER_LENGTH
    raw_text: str = ""

    def to_identifier(self, codebook: AttributeCodebook) -> list[int]:
        """KAE: identifier is a sequence of codebook indices, one per position."""
        return [codebook.encode(a) for a in self.attributes[:IDENTIFIER_LENGTH]]


# ---------------------------------------------------------------------------
# 3. Generative Retrieval model (toy seq2seq)
# ---------------------------------------------------------------------------

class ToySeq2SeqRetriever:
    """
    Minimal implementation of the generative retriever.
    In the real paper: a Transformer is trained with cross-entropy loss
    over KAE identifiers.  Here we use a lookup table for demonstration.
    """

    def __init__(self, codebook: AttributeCodebook) -> None:
        self.codebook = codebook
        # item_db: identifier_tuple → Item
        self._item_db: dict[tuple[int, ...], Item] = {}
        # query → attribute_tokens (simulated; in practice learned by seq2seq)
        self._query_index: dict[str, list[str]] = {}

    def index_item(self, item: Item) -> None:
        key = tuple(item.to_identifier(self.codebook))
        self._item_db[key] = item
        # Build toy inverted index: each attribute → item
        for attr in item.attributes:
            self._query_index.setdefault(attr, [])
            self._query_index[attr].append(item.item_id)

    def retrieve(self, query: str, top_k: int = 5) -> list[Item]:
        """
        Simulate greedy beam decoding: find items whose attributes overlap
        with query tokens.  In the full paper, this is autoregressive decoding.
        """
        query_tokens = query.lower().split()
        scores: dict[str, int] = {}
        for token in query_tokens:
            for item_id in self._query_index.get(token, []):
                scores[item_id] = scores.get(item_id, 0) + 1
        # Sort by overlap score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        result_ids = [r[0] for r in ranked[:top_k]]
        return [it for it in self._item_db.values() if it.item_id in result_ids]


# ---------------------------------------------------------------------------
# 4. End-to-end demo
# ---------------------------------------------------------------------------

def main() -> None:
    codebook = AttributeCodebook()
    retriever = ToySeq2SeqRetriever(codebook)

    # ---- Build toy item catalog ----
    items = [
        Item("A001", ["fashion", "shirt", "men", "blue"],   "Blue men's shirt"),
        Item("A002", ["fashion", "sneaker", "women", "red"],  "Red women's sneakers"),
        Item("A003", ["electronics", "phone", "premium", "black"], "Premium black phone"),
        Item("A004", ["beauty", "lipstick", "women", "red"], "Red lipstick"),
        Item("A005", ["sports", "sneaker", "men", "white"],  "White sports sneakers for men"),
    ]
    for item in items:
        retriever.index_item(item)

    # ---- Normal retrieval ----
    print("=== Normal retrieval ===")
    results = retriever.retrieve("men shirt", top_k=3)
    for r in results:
        print(f"  {r.item_id}: {r.raw_text}  attrs={r.attributes}")

    # ---- Reserved-slot binding: new keyword injected at runtime ----
    print("\n=== Binding new keyword to reserved slot (no retrain) ===")
    codebook.bind_reserved_slot(0, "2026summer")

    # Add a new-season item using the freshly bound keyword
    new_item = Item(
        "B001",
        ["fashion", "shirt", "men", "2026summer"],
        "2026 summer collection shirt",
    )
    retriever.index_item(new_item)

    results2 = retriever.retrieve("2026summer shirt", top_k=3)
    print("Query: '2026summer shirt'")
    for r in results2:
        print(f"  {r.item_id}: {r.raw_text}  attrs={r.attributes}")

    # ---- Show identifier encoding ----
    print("\n=== KAE identifier encoding ===")
    for item in items[:3]:
        ident = item.to_identifier(codebook)
        decoded = [codebook.decode(i) for i in ident]
        print(f"  {item.item_id}  →  indices={ident}  words={decoded}")


if __name__ == "__main__":
    main()
