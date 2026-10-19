from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import torch
import torch.nn as nn

from data import Product, QueryCase, bow


class NeuralRanker(nn.Module):
    """Small query-product ranker standing in for the paper's generative ranker."""

    def __init__(self, vocab_size: int, hidden: int = 64):
        super().__init__()
        self.query = nn.Linear(vocab_size, hidden)
        self.product = nn.Linear(vocab_size, hidden)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, q_bow: torch.Tensor, p_bow: torch.Tensor) -> torch.Tensor:
        q = torch.tanh(self.query(q_bow))
        p = torch.tanh(self.product(p_bow))
        return (q * p).sum(dim=-1) / self.temperature.clamp_min(0.1)

    @torch.no_grad()
    def rank(self, case: QueryCase, vocab: Dict[str, int]) -> List[str]:
        q = bow(case.query, vocab)
        scored = []
        for product in case.candidates:
            score = self(q, bow(product.as_text(), vocab)).item()
            scored.append((score, product.product_id))
        return [pid for _, pid in sorted(scored, reverse=True)]


@dataclass
class Behavior:
    behavior_id: str
    name: str
    instruction: str
    target_fields: Tuple[str, ...]
    triggers: Tuple[str, ...]
    insert_text: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    active: bool = True

    @property
    def applications(self) -> int:
        return self.helpful + self.harmful + self.neutral

    @property
    def score(self) -> int:
        return self.helpful - self.harmful


@dataclass
class BehaviorHandbook:
    behaviors: Dict[str, Behavior] = field(default_factory=dict)
    min_trials: int = 2

    def active_behaviors(self) -> List[Behavior]:
        return [b for b in self.behaviors.values() if b.active]

    def update_stats(self, applied: Iterable[str], gain: int):
        for bid in applied:
            behavior = self.behaviors[bid]
            if gain > 0:
                behavior.helpful += 1
            elif gain < 0:
                behavior.harmful += 1
            else:
                behavior.neutral += 1
            if behavior.applications >= self.min_trials and behavior.score < 0:
                behavior.active = False

    def add_or_modify_from_failures(self, failures: List[QueryCase]):
        """A compact Reflector + Moderator + Curator.

        It detects recurring uncovered query terms and adds a field-scoped
        behavior. Embedding/LLM dedup from the paper is approximated with
        trigger overlap to keep the reproduction self-contained.
        """
        known = {trigger for b in self.behaviors.values() for trigger in b.triggers}
        for case in failures:
            for token in case.query.lower().split():
                if len(token) < 6 or token in known:
                    continue
                new_id = f"B{len(self.behaviors) + 1:03d}"
                self.behaviors[new_id] = Behavior(
                    behavior_id=new_id,
                    name=f"Surface{token.title()}Evidence",
                    instruction=f"When the query mentions {token}, surface exact listing evidence near the top.",
                    target_fields=("features", "description"),
                    triggers=(token,),
                    insert_text=f"Query-matched {token} evidence",
                )
                known.add(token)
                return


class RuleBasedEditor:
    """Field-scoped editor that emulates the LLM product editor contract."""

    def rewrite(self, product: Product, query: str, handbook: BehaviorHandbook) -> Tuple[Product, List[str]]:
        updates: Dict[str, str] = {}
        applied: List[str] = []
        lower_query = query.lower()
        original = {field: getattr(product, field) for field in ("title", "features", "description")}

        for behavior in handbook.active_behaviors():
            if not any(trigger in lower_query for trigger in behavior.triggers):
                continue
            applied.append(behavior.behavior_id)
            for field in behavior.target_fields:
                current = updates.get(field, original[field])
                evidence_words = [w for w in lower_query.split() if w in product.as_text().lower()]
                evidence = ", ".join(evidence_words[:3]) or behavior.insert_text
                addition = f"{behavior.insert_text}: {evidence}."
                if addition.lower() not in current.lower():
                    updates[field] = f"{addition} {current}"

        return product.copy_with(updates), applied


def initial_handbook() -> BehaviorHandbook:
    return BehaviorHandbook(
        behaviors={
            "B001": Behavior("B001", "NameExactProductType", "Put the exact product type early.", ("title", "description"), ("compressor", "cable", "boots", "grinder"), "Exact product-type match"),
            "B002": Behavior("B002", "MapFailureModeToMaterial", "Map durability concerns to material/build evidence.", ("features", "description"), ("break", "durable", "stitched"), "Durability evidence"),
            "B003": Behavior("B003", "SurfaceUseCase", "Place the user use-case in the opening description.", ("description",), ("car", "airbrush", "french", "work"), "Use-case fit"),
            "B004": Behavior("B004", "SurfaceConstraint", "Name exact query constraints supported by the listing.", ("features",), ("quiet", "waterproof", "coarse", "quick"), "Constraint match"),
        }
    )
