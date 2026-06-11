"""
Toy e-commerce dataset for QueryAgent-R1 reproduction.

Provides:
  - ProductCatalog: searchable product inventory (FAISS-backed dense retrieval)
  - UserHistoryDataset: user interaction sequences for interest graph construction
  - QueryDataset: (user_id, candidate_queries, labels) for offline evaluation

Interface mirrors the paper's data flow:
  user history → memory abstraction → interest graph
  query candidates → retrieval validation → consistency reward
"""

import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


CATEGORIES = ["clothing", "electronics", "beauty", "sports", "home", "food"]

PRODUCT_TEMPLATES = [
    "{color} {category} {style} for {gender}",
    "{brand} {category} {size} {style}",
    "premium {category} {style} {material}",
    "{season} {category} collection {style}",
]

COLORS = ["red", "blue", "black", "white", "green", "navy", "grey"]
STYLES = ["casual", "formal", "sport", "vintage", "modern", "slim-fit"]
GENDERS = ["men", "women", "unisex"]
BRANDS = ["AlphaX", "NovaCo", "ZenBrand", "PureStyle", "UrbanEdge"]
SIZES = ["S", "M", "L", "XL", "XXL"]
MATERIALS = ["cotton", "polyester", "leather", "silk", "denim"]
SEASONS = ["summer", "winter", "spring", "autumn", "all-season"]


@dataclass
class Product:
    id: int
    title: str
    category: str
    price: float
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_text(self) -> str:
        return f"{self.title} | category: {self.category} | price: ${self.price:.2f}"


@dataclass
class UserHistory:
    user_id: int
    clicked_product_ids: List[int]
    purchased_product_ids: List[int]
    query_history: List[str]


def _make_product(pid: int, rng: random.Random) -> Product:
    cat = rng.choice(CATEGORIES)
    template = rng.choice(PRODUCT_TEMPLATES)
    title = template.format(
        color=rng.choice(COLORS),
        category=cat,
        style=rng.choice(STYLES),
        gender=rng.choice(GENDERS),
        brand=rng.choice(BRANDS),
        size=rng.choice(SIZES),
        material=rng.choice(MATERIALS),
        season=rng.choice(SEASONS),
    )
    price = round(rng.uniform(9.99, 299.99), 2)
    return Product(id=pid, title=title, category=cat, price=price)


class ProductCatalog:
    """Searchable product catalog with dense retrieval (toy FAISS index)."""

    def __init__(self, n_products: int = 5000, embedding_dim: int = 64, seed: int = 42):
        self.embedding_dim = embedding_dim
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        self.products: List[Product] = []
        for i in range(n_products):
            p = _make_product(i, rng)
            p.embedding = np_rng.standard_normal(embedding_dim).astype(np.float32)
            p.embedding /= np.linalg.norm(p.embedding) + 1e-8
            self.products.append(p)

        # Build FAISS-style index (pure numpy for toy purposes)
        self._embeddings = np.stack([p.embedding for p in self.products])  # (N, D)

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Product]:
        """Retrieve top-k products by cosine similarity."""
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = self._embeddings @ q  # (N,)
        top_ids = np.argpartition(scores, -top_k)[-top_k:]
        top_ids = top_ids[np.argsort(-scores[top_ids])]
        return [self.products[i] for i in top_ids]

    def get_by_ids(self, ids: List[int]) -> List[Product]:
        return [self.products[i] for i in ids if i < len(self.products)]

    def __len__(self) -> int:
        return len(self.products)


class UserHistoryDataset:
    """Synthetic user interaction histories for interest graph construction."""

    def __init__(
        self,
        catalog: ProductCatalog,
        n_users: int = 1000,
        seq_len: int = 20,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.users: Dict[int, UserHistory] = {}

        query_vocab = [
            f"{s} {c}" for s in STYLES for c in CATEGORIES
        ] + [f"{b} {c}" for b in BRANDS for c in CATEGORIES]

        for uid in range(n_users):
            # Users have a dominant category preference
            pref_cat = rng.choice(CATEGORIES)
            pref_products = [p for p in catalog.products if p.category == pref_cat]
            if not pref_products:
                pref_products = catalog.products

            clicked = [rng.choice(pref_products).id for _ in range(seq_len)]
            purchased = [rng.choice(clicked) for _ in range(min(3, seq_len // 5))]
            queries = [rng.choice(query_vocab) for _ in range(seq_len // 4)]

            self.users[uid] = UserHistory(
                user_id=uid,
                clicked_product_ids=clicked,
                purchased_product_ids=purchased,
                query_history=queries,
            )

    def get_user(self, user_id: int) -> Optional[UserHistory]:
        return self.users.get(user_id)

    def __len__(self) -> int:
        return len(self.users)


@dataclass
class QuerySample:
    user_id: int
    candidate_queries: List[str]
    positive_query: str          # ground-truth high-CVR query
    retrieved_product_ids: List[int]  # products retrieved by the positive query


class QueryDataset:
    """
    Offline evaluation dataset: (user, candidate queries, ground-truth label).
    Simulates the paper's two dataset splits (proprietary + public proxy).
    """

    def __init__(
        self,
        user_dataset: UserHistoryDataset,
        catalog: ProductCatalog,
        n_samples: int = 2000,
        n_candidates: int = 5,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.samples: List[QuerySample] = []
        query_pool = [
            f"{s} {c}" for s in STYLES for c in CATEGORIES
        ]

        for _ in range(n_samples):
            uid = rng.randint(0, len(user_dataset) - 1)
            candidates = [rng.choice(query_pool) for _ in range(n_candidates)]
            positive = rng.choice(candidates)
            # Simulate retrieved products for positive query
            q_emb = np.random.default_rng(abs(hash(positive)) % (2**31)).standard_normal(
                catalog.embedding_dim
            ).astype(np.float32)
            retrieved = [p.id for p in catalog.search(q_emb, top_k=5)]
            self.samples.append(
                QuerySample(
                    user_id=uid,
                    candidate_queries=candidates,
                    positive_query=positive,
                    retrieved_product_ids=retrieved,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> QuerySample:
        return self.samples[idx]


def build_datasets(
    n_products: int = 5000,
    n_users: int = 1000,
    n_train: int = 1600,
    n_eval: int = 400,
    embedding_dim: int = 64,
    seed: int = 42,
) -> Tuple[ProductCatalog, UserHistoryDataset, QueryDataset, QueryDataset]:
    catalog = ProductCatalog(n_products=n_products, embedding_dim=embedding_dim, seed=seed)
    users = UserHistoryDataset(catalog=catalog, n_users=n_users, seed=seed)
    train_set = QueryDataset(users, catalog, n_samples=n_train, seed=seed)
    eval_set = QueryDataset(users, catalog, n_samples=n_eval, seed=seed + 1)
    return catalog, users, train_set, eval_set


if __name__ == "__main__":
    catalog, users, train_set, eval_set = build_datasets()
    print(f"Products: {len(catalog)}, Users: {len(users)}")
    print(f"Train samples: {len(train_set)}, Eval samples: {len(eval_set)}")
    s = train_set[0]
    print(f"Sample user={s.user_id}, candidates={s.candidate_queries[:2]}, positive='{s.positive_query}'")
