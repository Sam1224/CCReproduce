from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

ATTRS: Dict[str, Sequence[str]] = {
    "color": ["red", "blue", "black", "white", "green", "silver"],
    "pattern": ["solid", "floral", "striped", "polka dot", "plaid"],
    "sleeve": ["sleeveless", "short sleeve", "long sleeve"],
    "neckline": ["v-neck", "round neck", "collared"],
    "shape": ["fit-and-flare", "straight", "oversized", "slim"],
    "material": ["cotton", "denim", "silk", "knit"],
}


@dataclass(frozen=True)
class Product:
    product_id: int
    attrs: Dict[str, str]

    @property
    def title(self) -> str:
        ordered = [
            self.attrs["color"],
            self.attrs["pattern"],
            self.attrs["shape"],
            self.attrs["material"],
            self.attrs["sleeve"],
            self.attrs["neckline"],
            "dress",
        ]
        return " ".join(ordered)


@dataclass(frozen=True)
class Query:
    reference_id: int
    target_id: int
    modification: str
    changed_attrs: Dict[str, str]


def build_catalog(seed: int = 7, max_items: int = 720) -> List[Product]:
    rng = random.Random(seed)
    all_products: List[Product] = []
    pid = 0
    values = list(ATTRS.items())
    for color in values[0][1]:
        for pattern in values[1][1]:
            for sleeve in values[2][1]:
                for neckline in values[3][1]:
                    for shape in values[4][1]:
                        for material in values[5][1]:
                            all_products.append(
                                Product(
                                    pid,
                                    {
                                        "color": color,
                                        "pattern": pattern,
                                        "sleeve": sleeve,
                                        "neckline": neckline,
                                        "shape": shape,
                                        "material": material,
                                    },
                                )
                            )
                            pid += 1
    rng.shuffle(all_products)
    return [Product(i, p.attrs) for i, p in enumerate(all_products[:max_items])]


def _find_product(catalog: Sequence[Product], attrs: Dict[str, str]) -> Product | None:
    for product in catalog:
        if product.attrs == attrs:
            return product
    return None


def generate_queries(
    catalog: Sequence[Product],
    seed: int = 13,
    n_queries: int = 300,
    changes_per_query: Tuple[int, int] = (1, 2),
) -> List[Query]:
    rng = random.Random(seed)
    queries: List[Query] = []
    attempts = 0
    while len(queries) < n_queries and attempts < n_queries * 50:
        attempts += 1
        reference = rng.choice(catalog)
        attrs = dict(reference.attrs)
        n_changes = rng.randint(*changes_per_query)
        keys = rng.sample(list(ATTRS.keys()), n_changes)
        changed: Dict[str, str] = {}
        for key in keys:
            choices = [v for v in ATTRS[key] if v != attrs[key]]
            new_value = rng.choice(choices)
            attrs[key] = new_value
            changed[key] = new_value
        target = _find_product(catalog, attrs)
        if target is None or target.product_id == reference.product_id:
            continue
        modification = "make it " + " and ".join(changed.values())
        queries.append(Query(reference.product_id, target.product_id, modification, changed))
    return queries


def iter_titles(catalog: Iterable[Product]) -> Iterable[str]:
    for product in catalog:
        yield product.title
