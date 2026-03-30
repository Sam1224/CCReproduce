from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class WebSpec:
    has_nav: bool
    has_search: bool
    has_footer: bool
    num_cards: int

    def to_features(self) -> List[float]:
        # Keep features tiny on purpose.
        return [
            1.0 if self.has_nav else 0.0,
            1.0 if self.has_search else 0.0,
            1.0 if self.has_footer else 0.0,
            float(self.num_cards) / 10.0,
        ]


def spec_from_json(obj: Dict) -> WebSpec:
    return WebSpec(
        has_nav=bool(obj["has_nav"]),
        has_search=bool(obj["has_search"]),
        has_footer=bool(obj["has_footer"]),
        num_cards=int(obj["num_cards"]),
    )
