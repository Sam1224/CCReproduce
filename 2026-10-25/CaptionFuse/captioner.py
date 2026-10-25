from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from dataset import Product


class Captioner(ABC):
    """VLM captioning abstraction used by CaptionFuse."""

    @abstractmethod
    def generate(self, reference: Product, modification: str, changed_attrs: Dict[str, str]) -> str:
        """Return an attribute-rich product-title caption for the target item."""


class RuleBasedCaptioner(Captioner):
    """A deterministic stand-in for a VLM.

    The real paper asks a VLM to infer target attributes from the reference image and
    the user's sparse modification. In this toy pipeline the reference attributes are
    already structured, so we can deterministically emulate the same transformation.
    """

    def generate(self, reference: Product, modification: str, changed_attrs: Dict[str, str]) -> str:
        attrs = dict(reference.attrs)
        attrs.update(changed_attrs)
        return " ".join(
            [
                attrs["color"],
                attrs["pattern"],
                attrs["shape"],
                attrs["material"],
                attrs["sleeve"],
                attrs["neckline"],
                "dress",
            ]
        )


def should_route_to_captioner(modification: str, min_attribute_words: int = 4) -> bool:
    """Simple hybrid-routing heuristic from the paper's cost-saving idea.

    If the user modification is already descriptive enough, one may bypass the VLM.
    This toy heuristic counts content words and routes short/sparse modifications to
    the captioner.
    """

    tokens = [t for t in modification.replace("-", " ").split() if len(t) > 2]
    return len(tokens) < min_attribute_words
