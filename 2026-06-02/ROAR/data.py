import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Example:
    query: str
    product: str


@dataclass
class GroupExample:
    query: str
    products: List[str]
    # products are ordered by relevance: best -> worst


@dataclass
class DatasetBundle:
    stage1_train: List[Example]
    stage2_train: List[GroupExample]
    eval_groups: List[GroupExample]
    catalog: List[str]


def _make_product(brand: str, category: str, attr: str) -> str:
    # Keep text format similar to e-com product title
    return f"{brand} {category} {attr}".strip()


def build_toy_ecommerce_dataset(
    seed: int = 13,
    n_brands: int = 10,
    n_categories: int = 6,
    n_attrs: int = 12,
    n_queries: int = 2000,
    group_size_range: Tuple[int, int] = (2, 7),
) -> DatasetBundle:
    """Create a small synthetic ESCI-like dataset.

    The real paper uses platform logs + human judgments with a 4-grade ordering
    (Exact > Substitute > Complementary > Irrelevant). We mimic the *interfaces*
    and the ordering with controllable noise.
    """

    rng = random.Random(seed)

    brands = [f"Brand{idx}" for idx in range(n_brands)]
    categories = [
        "milk",
        "shampoo",
        "phone",
        "laptop",
        "shoe",
        "tshirt",
    ][:n_categories]
    attrs = [f"Attr{idx}" for idx in range(n_attrs)]

    # Build catalog
    catalog: List[str] = []
    for brand in brands:
        for category in categories:
            for attr in attrs[: max(3, n_attrs // 3)]:
                catalog.append(_make_product(brand, category, attr))

    # Helpers
    def pick_other_brand(brand: str) -> str:
        other = [b for b in brands if b != brand]
        return rng.choice(other)

    def pick_other_category(category: str) -> str:
        other = [c for c in categories if c != category]
        return rng.choice(other)

    def pick_other_attr(attr: str) -> str:
        other = [a for a in attrs if a != attr]
        return rng.choice(other)

    # Category-level complements (roughly)
    complements: Dict[str, List[str]] = {
        "milk": ["tshirt", "shoe"],
        "shampoo": ["tshirt"],
        "phone": ["laptop"],
        "laptop": ["phone"],
        "shoe": ["tshirt"],
        "tshirt": ["shoe"],
    }

    stage1_train: List[Example] = []
    stage2_train: List[GroupExample] = []
    eval_groups: List[GroupExample] = []

    for idx in range(n_queries):
        brand = rng.choice(brands)
        category = rng.choice(categories)
        attr = rng.choice(attrs)

        # Query: colloquial-ish pattern
        query = rng.choice(
            [
                f"{brand} {category}",
                f"buy {brand} {category}",
                f"{brand} {category} {attr}",
                f"need {category} {brand}",
                f"{category} {attr} {brand}",
            ]
        )

        exact = _make_product(brand, category, attr)
        substitute = _make_product(pick_other_brand(brand), category, attr)

        comp_cat = rng.choice(complements.get(category, categories))
        complementary = _make_product(brand, comp_cat, pick_other_attr(attr))

        irrelevant = _make_product(pick_other_brand(brand), pick_other_category(category), pick_other_attr(attr))

        # Stage1: use (query, exact) pairs, sometimes treat substitute as positive
        # for branded queries to mimic the paper's "substitute positives" option.
        if rng.random() < 0.2:
            pos = substitute
        else:
            pos = exact
        stage1_train.append(Example(query=query, product=pos))

        # Stage2: build a variable-size ordered group
        group_size = rng.randint(group_size_range[0], group_size_range[1])
        group = [exact, substitute, complementary, irrelevant]
        # Fill with more negatives / near-duplicates
        while len(group) < group_size:
            if rng.random() < 0.5:
                # near-duplicate substitute variants
                group.append(_make_product(pick_other_brand(brand), category, pick_other_attr(attr)))
            else:
                group.append(_make_product(pick_other_brand(brand), pick_other_category(category), pick_other_attr(attr)))

        # Ensure the first four follow the core ordering, and then append rest as worst
        ordered = group[:3] + [irrelevant] + group[4:]
        stage2_train.append(GroupExample(query=query, products=ordered[:group_size]))

        # Hold-out for evaluation (roughly 10%)
        if idx % 10 == 0:
            eval_groups.append(GroupExample(query=query, products=ordered[:group_size]))

    return DatasetBundle(
        stage1_train=stage1_train,
        stage2_train=stage2_train,
        eval_groups=eval_groups,
        catalog=catalog,
    )
