from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class Catalog:
    item_features: torch.Tensor
    item_categories: torch.Tensor
    popularity: torch.Tensor
    main_codes: torch.Tensor
    aux_codes: torch.Tensor
    codebook_size: int
    history_len: int

    def __post_init__(self) -> None:
        self.num_items = int(self.item_features.size(0))
        self.feature_dim = int(self.item_features.size(1))
        self.main_lookup: Dict[Tuple[int, int], int] = {}
        self.aux_lookup: Dict[Tuple[int, int], int] = {}
        for item_id in range(self.num_items):
            main_key = tuple(int(v) for v in self.main_codes[item_id].tolist())
            aux_key = tuple(int(v) for v in self.aux_codes[item_id].tolist())
            self.main_lookup[main_key] = item_id
            self.aux_lookup[aux_key] = item_id

    def resolve_main(self, codes: Iterable[int]) -> int | None:
        return self.main_lookup.get(tuple(int(v) for v in codes))

    def resolve_aux(self, codes: Iterable[int]) -> int | None:
        return self.aux_lookup.get(tuple(int(v) for v in codes))


class ToyBARGEDataset(Dataset):
    def __init__(self, history: torch.Tensor, target_items: torch.Tensor, catalog: Catalog) -> None:
        self.history = history.long()
        self.target_items = target_items.long()
        self.catalog = catalog

    def __len__(self) -> int:
        return int(self.history.size(0))

    def __getitem__(self, index: int):
        history = self.history[index]
        target_item = self.target_items[index]
        return {
            "history": history,
            "target_item": target_item,
            "main_codes": self.catalog.main_codes[target_item],
            "aux_codes": self.catalog.aux_codes[target_item],
        }


def build_catalog(codebook_size: int = 6, feature_dim: int = 16, history_len: int = 6, seed: int = 7) -> Catalog:
    generator = torch.Generator().manual_seed(seed)
    num_items = codebook_size * codebook_size

    item_ids = torch.arange(num_items)
    main_first = item_ids // codebook_size
    main_second = item_ids % codebook_size
    main_codes = torch.stack([main_first, main_second], dim=1)

    aux_first = main_second
    aux_second = (main_first + main_second) % codebook_size
    aux_codes = torch.stack([aux_first, aux_second], dim=1)

    category_proto = torch.randn(codebook_size, feature_dim, generator=generator)
    local_proto = torch.randn(codebook_size, feature_dim, generator=generator)

    features = []
    popularity = []
    categories = []
    for item_id in range(num_items):
        c1 = int(main_first[item_id])
        c2 = int(main_second[item_id])
        feat = 1.2 * category_proto[c1] + 0.8 * local_proto[c2]
        feat = feat + 0.05 * torch.randn(feature_dim, generator=generator)
        features.append(feat)
        popularity.append(float(codebook_size - c2) / codebook_size)
        categories.append(c1)

    return Catalog(
        item_features=torch.stack(features, dim=0),
        item_categories=torch.tensor(categories, dtype=torch.long),
        popularity=torch.tensor(popularity, dtype=torch.float32),
        main_codes=main_codes.long(),
        aux_codes=aux_codes.long(),
        codebook_size=codebook_size,
        history_len=history_len,
    )


def _pick_history_items(
    catalog: Catalog,
    dominant_group: int,
    secondary_group: int,
    history_len: int,
    generator: torch.Generator,
) -> List[int]:
    dominant_items = [idx for idx, g in enumerate(catalog.item_categories.tolist()) if g == dominant_group]
    secondary_items = [idx for idx, g in enumerate(catalog.item_categories.tolist()) if g == secondary_group]
    history: List[int] = []
    for step in range(history_len):
        pool = dominant_items if step < history_len - 2 else secondary_items
        weights = torch.tensor([catalog.popularity[i] for i in pool], dtype=torch.float32)
        choice = int(torch.multinomial(weights, num_samples=1, generator=generator).item())
        history.append(pool[choice])
    return history


def _pick_target_item(catalog: Catalog, history: List[int], dominant_group: int, generator: torch.Generator) -> int:
    user_profile = catalog.item_features[history].mean(dim=0)
    scores = catalog.item_features @ user_profile
    mask = catalog.item_categories == dominant_group
    for item_id in history:
        mask[item_id] = False
    scores = scores.masked_fill(~mask, -1e9)
    noise = 0.02 * torch.randn(scores.shape, generator=generator, device=scores.device)
    return int((scores + noise).argmax().item())


def create_dataset(
    num_samples: int = 640,
    codebook_size: int = 6,
    feature_dim: int = 16,
    history_len: int = 6,
    seed: int = 7,
) -> Tuple[Catalog, ToyBARGEDataset]:
    catalog = build_catalog(codebook_size=codebook_size, feature_dim=feature_dim, history_len=history_len, seed=seed)
    generator = torch.Generator().manual_seed(seed + 13)

    histories = []
    targets = []
    for _ in range(num_samples):
        dominant = int(torch.randint(codebook_size, (1,), generator=generator).item())
        secondary = int((dominant + torch.randint(1, codebook_size, (1,), generator=generator).item()) % codebook_size)
        history = _pick_history_items(catalog, dominant, secondary, history_len, generator)
        target_item = _pick_target_item(catalog, history, dominant, generator)
        histories.append(torch.tensor(history, dtype=torch.long))
        targets.append(target_item)

    dataset = ToyBARGEDataset(torch.stack(histories), torch.tensor(targets, dtype=torch.long), catalog)
    return catalog, dataset


def _collate(batch):
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def create_dataloaders(
    batch_size: int = 32,
    num_samples: int = 640,
    codebook_size: int = 6,
    feature_dim: int = 16,
    history_len: int = 6,
    seed: int = 7,
):
    catalog, dataset = create_dataset(
        num_samples=num_samples,
        codebook_size=codebook_size,
        feature_dim=feature_dim,
        history_len=history_len,
        seed=seed,
    )
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed + 99)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return catalog, train_loader, val_loader
