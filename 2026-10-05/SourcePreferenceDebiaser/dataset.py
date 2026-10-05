import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


SOURCES = ["amazon", "etsy", "shopify", "booking", "expedia", "semantic_scholar", "arxiv", "github_pages"]
PREFERRED = {"amazon", "booking", "semantic_scholar", "github_pages"}
DOMAINS = ["shopping", "accommodation", "scholarly"]


@dataclass
class PairExample:
    left_features: torch.Tensor
    right_features: torch.Tensor
    left_source: int
    right_source: int
    label: int
    domain: int


def make_item(requirements_met: int, position: int, source: str, domain: str) -> Tuple[torch.Tensor, int, int]:
    source_bias = 1.0 if source in PREFERRED else 0.0
    normalized_requirements = requirements_met / 5.0
    normalized_position = 1.0 - position / 10.0
    features = torch.tensor(
        [normalized_requirements, normalized_position, source_bias], dtype=torch.float32
    )
    return features, SOURCES.index(source), DOMAINS.index(domain)


class SourcePreferenceDataset(Dataset):
    def __init__(self, size: int = 2000, seed: int = 7, bias_strength: float = 0.9):
        self.examples: List[PairExample] = []
        rng = random.Random(seed)
        for _ in range(size):
            domain = rng.choice(DOMAINS)
            left_source = rng.choice(SOURCES)
            right_source = rng.choice([s for s in SOURCES if s != left_source])
            left_req = rng.randint(1, 5)
            delta = rng.choice([-1, 0, 1])
            right_req = max(1, min(5, left_req + delta))
            position = rng.randint(1, 10)
            left_features, left_sid, did = make_item(left_req, position, left_source, domain)
            right_features, right_sid, _ = make_item(right_req, position, right_source, domain)
            utility_margin = float(left_req - right_req)
            source_margin = (1.0 if left_source in PREFERRED else -1.0) - (1.0 if right_source in PREFERRED else -1.0)
            noisy_score = utility_margin + bias_strength * source_margin + rng.gauss(0, 0.2)
            label = 1 if noisy_score >= 0 else 0
            self.examples.append(PairExample(left_features, right_features, left_sid, right_sid, label, did))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[index]
        return {
            "left_features": ex.left_features,
            "right_features": ex.right_features,
            "left_source": torch.tensor(ex.left_source, dtype=torch.long),
            "right_source": torch.tensor(ex.right_source, dtype=torch.long),
            "label": torch.tensor(ex.label, dtype=torch.float32),
            "domain": torch.tensor(ex.domain, dtype=torch.long),
        }


def build_counterfactual_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    swapped = dict(batch)
    swapped["left_source"] = batch["right_source"]
    swapped["right_source"] = batch["left_source"]
    return swapped
