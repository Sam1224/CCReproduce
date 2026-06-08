"""
Toy e-commerce dataset for QueryAgent-R1 reproduction.
Provides the data interface compatible with the paper's framework.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import random


# --- Toy product catalog ---
TOY_PRODUCTS = [
    {"id": "p001", "text": "wireless bluetooth headphones noise cancelling premium", "category": "electronics"},
    {"id": "p002", "text": "running shoes lightweight breathable marathon", "category": "sports"},
    {"id": "p003", "text": "coffee maker automatic espresso machine kitchen", "category": "kitchen"},
    {"id": "p004", "text": "yoga mat non slip thick extra wide meditation", "category": "sports"},
    {"id": "p005", "text": "laptop backpack waterproof usb charging port", "category": "bags"},
    {"id": "p006", "text": "smartwatch fitness tracker heart rate GPS", "category": "electronics"},
    {"id": "p007", "text": "protein powder whey vanilla muscle recovery", "category": "health"},
    {"id": "p008", "text": "air purifier HEPA filter quiet bedroom", "category": "home"},
    {"id": "p009", "text": "mechanical keyboard RGB gaming programmable", "category": "electronics"},
    {"id": "p010", "text": "vitamin D3 supplement immune support 5000 IU", "category": "health"},
    {"id": "p011", "text": "hiking boots waterproof trail ankle support", "category": "sports"},
    {"id": "p012", "text": "blender high speed smoothie frozen fruit", "category": "kitchen"},
    {"id": "p013", "text": "resistance bands set workout home gym", "category": "sports"},
    {"id": "p014", "text": "portable charger 20000mAh USB-C fast charge", "category": "electronics"},
    {"id": "p015", "text": "standing desk adjustable height electric", "category": "furniture"},
    {"id": "p016", "text": "face serum vitamin C brightening anti-aging", "category": "beauty"},
    {"id": "p017", "text": "insulated water bottle stainless steel 32oz", "category": "sports"},
    {"id": "p018", "text": "noise machine sleep white noise relaxation", "category": "health"},
    {"id": "p019", "text": "plant based protein powder vegan chocolate", "category": "health"},
    {"id": "p020", "text": "ergonomic office chair lumbar support mesh", "category": "furniture"},
]

# Toy vocabulary
VOCAB = ["<pad>", "<bos>", "<eos>", "<unk>"] + list(
    set(
        " ".join(p["text"] for p in TOY_PRODUCTS).split()
        + "wireless headphones running shoes coffee yoga laptop smart protein air keyboard".split()
    )
)
TOKEN2ID = {tok: i for i, tok in enumerate(VOCAB)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = TOKEN2ID["<pad>"]
BOS_ID = TOKEN2ID["<bos>"]
EOS_ID = TOKEN2ID["<eos>"]
UNK_ID = TOKEN2ID["<unk>"]


def tokenize(text: str, max_len: int = 16) -> List[int]:
    tokens = [BOS_ID]
    for word in text.lower().split():
        tokens.append(TOKEN2ID.get(word, UNK_ID))
        if len(tokens) >= max_len - 1:
            break
    tokens.append(EOS_ID)
    return tokens


def pad_sequence(seq: List[int], max_len: int) -> Tuple[List[int], List[int]]:
    mask = [1] * len(seq) + [0] * (max_len - len(seq))
    seq = seq + [PAD_ID] * (max_len - len(seq))
    return seq[:max_len], mask[:max_len]


class SimpleTokenizer:
    def __init__(self):
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.pad_token_id = PAD_ID
        self.vocab_size = VOCAB_SIZE

    def encode(self, text: str, max_len: int = 16) -> List[int]:
        return tokenize(text, max_len)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special = {PAD_ID, BOS_ID, EOS_ID} if skip_special_tokens else set()
        return " ".join(ID2TOKEN.get(i, "<unk>") for i in ids if i not in special)


class ECommerceQueryDataset(Dataset):
    """
    Toy dataset of (user_history, target_query, converted_product) triples.
    Interface-aligned with the paper's data format.
    """

    def __init__(
        self,
        num_users: int = 200,
        history_len: int = 20,
        item_embed_dim: int = 64,
        max_query_len: int = 16,
        seed: int = 42,
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        self.item_embed_dim = item_embed_dim
        self.max_query_len = max_query_len
        self.tokenizer = SimpleTokenizer()

        # Pre-compute product embeddings (toy: random but consistent)
        self.product_embeds = torch.randn(len(TOY_PRODUCTS), item_embed_dim)
        self.product_embeds = torch.nn.functional.normalize(self.product_embeds, dim=-1)

        # Generate user records
        self.records = []
        for user_id in range(num_users):
            # Random user preference cluster
            pref_category = random.choice(["electronics", "sports", "health", "kitchen"])
            preferred_ids = [
                i for i, p in enumerate(TOY_PRODUCTS) if p["category"] == pref_category
            ]
            other_ids = [i for i in range(len(TOY_PRODUCTS)) if i not in preferred_ids]

            # History: mostly preferred + some random
            hist_indices = random.choices(preferred_ids, k=max(1, history_len - 5))
            hist_indices += random.choices(other_ids or [0], k=min(5, history_len - len(hist_indices)))
            random.shuffle(hist_indices)
            hist_indices = hist_indices[:history_len]

            # Target query: from a preferred product's text
            target_prod_idx = random.choice(preferred_ids)
            target_prod = TOY_PRODUCTS[target_prod_idx]
            target_query = " ".join(target_prod["text"].split()[:4])  # first 4 words

            self.records.append({
                "user_id": user_id,
                "hist_indices": hist_indices,
                "target_query": target_query,
                "target_prod_idx": target_prod_idx,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # User history embeddings
        hist_embeds = self.product_embeds[rec["hist_indices"]]  # (L, D)
        hist_mask = torch.ones(len(rec["hist_indices"]))

        # Pad/truncate to fixed length
        L = 20
        if hist_embeds.shape[0] < L:
            pad = torch.zeros(L - hist_embeds.shape[0], self.item_embed_dim)
            hist_embeds = torch.cat([hist_embeds, pad], dim=0)
            hist_mask = torch.cat([hist_mask, torch.zeros(L - len(rec["hist_indices"]))])

        # Tokenize target query (teacher forcing input)
        query_ids = tokenize(rec["target_query"], self.max_query_len)
        query_ids, query_mask = pad_sequence(query_ids, self.max_query_len)

        # Ground-truth converted product
        pos_prod_embed = self.product_embeds[rec["target_prod_idx"]]

        return {
            "hist_embeds": hist_embeds,                        # (L, D)
            "hist_mask": hist_mask,                            # (L,)
            "query_ids": torch.tensor(query_ids),             # (T,)
            "query_mask": torch.tensor(query_mask),           # (T,)
            "pos_prod_embed": pos_prod_embed,                  # (D,)
            "target_query": rec["target_query"],
        }


def get_dataloaders(
    num_users: int = 200,
    history_len: int = 20,
    item_embed_dim: int = 64,
    batch_size: int = 16,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    dataset = ECommerceQueryDataset(
        num_users=num_users,
        history_len=history_len,
        item_embed_dim=item_embed_dim,
    )
    n_train = int(len(dataset) * train_ratio)
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_product_corpus() -> List[Dict]:
    return TOY_PRODUCTS
