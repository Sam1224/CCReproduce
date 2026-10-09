from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Product:
    item_id: int
    title: str
    brand: str
    category: str
    style: str
    material: str
    color: str
    price_bucket: str


@dataclass(frozen=True)
class UserExample:
    history_ids: List[int]
    positive_id: int
    query_text: str


@dataclass
class PreparedData:
    products: List[Product]
    train_examples: List[UserExample]
    val_examples: List[UserExample]
    test_examples: List[UserExample]
    vocab: Dict[str, int]
    pad_id: int
    unk_id: int
    max_query_len: int
    max_meta_len: int
    card_tensors: torch.Tensor
    image_tensors: torch.Tensor
    meta_tokens: torch.Tensor
    meta_masks: torch.Tensor


class SimpleTokenizer:
    def __init__(self, vocab: Dict[str, int], pad_id: int, unk_id: int) -> None:
        self.vocab = vocab
        self.pad_id = pad_id
        self.unk_id = unk_id

    def encode(self, text: str, max_len: int) -> Tuple[List[int], List[int]]:
        tokens = _normalize(text).split()
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens[:max_len]]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids.extend([self.pad_id] * pad_n)
            mask.extend([0] * pad_n)
        return ids, mask


def _normalize(text: str) -> str:
    return (
        text.lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(":", " ")
    )


def _make_base_image(product: Product, size: int = 128) -> Image.Image:
    color_map = {
        "green": (77, 166, 99),
        "blue": (79, 129, 189),
        "red": (201, 74, 74),
        "black": (55, 55, 55),
        "white": (225, 225, 225),
        "brown": (145, 102, 74),
        "gold": (196, 166, 84),
        "pink": (214, 122, 170),
        "gray": (139, 147, 157),
    }
    bg = color_map.get(product.color, (120, 120, 120))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    pad = size // 8
    center = size // 2
    shape_fill = tuple(max(0, c - 35) for c in bg)

    if product.category in {"pin", "brooch"}:
        draw.ellipse((pad, pad, size - pad, size - pad), fill=shape_fill)
    elif product.category in {"earrings"}:
        draw.rectangle((pad, pad, center - 6, size - pad), fill=shape_fill)
        draw.rectangle((center + 6, pad, size - pad, size - pad), fill=shape_fill)
    elif product.category in {"bag"}:
        draw.rounded_rectangle((pad, center - 24, size - pad, size - pad), radius=12, fill=shape_fill)
        draw.arc((center - 28, 12, center + 28, 58), start=180, end=360, fill=(245, 245, 245), width=4)
    elif product.category in {"shoes"}:
        draw.polygon(
            [(pad, center + 18), (size - pad, center + 18), (size - 2 * pad, size - pad), (pad + 18, size - pad)],
            fill=shape_fill,
        )
    else:
        draw.rounded_rectangle((pad, pad, size - pad, size - pad), radius=14, fill=shape_fill)

    font = ImageFont.load_default()
    label = product.category[:10].upper()
    tw = draw.textlength(label, font=font)
    draw.text(((size - tw) / 2, size - 18), label, fill=(250, 250, 250), font=font)
    return img


def render_product_card(product: Product, canvas_size: int = 224, image_height: int = 128) -> Image.Image:
    card = Image.new("RGB", (canvas_size, canvas_size), (248, 247, 244))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()

    base = _make_base_image(product, size=image_height)
    card.paste(base, ((canvas_size - image_height) // 2, 8))

    draw.rounded_rectangle((6, 6, canvas_size - 6, canvas_size - 6), radius=12, outline=(188, 188, 188), width=2)

    lines = [
        product.title[:28],
        f"brand: {product.brand}",
        f"cat: {product.category} | {product.style}",
        f"mat: {product.material} | {product.color}",
        f"price: {product.price_bucket}",
    ]
    y = image_height + 18
    for idx, line in enumerate(lines):
        fill = (28, 28, 28) if idx == 0 else (60, 60, 60)
        draw.text((12, y), line, fill=fill, font=font)
        y += 16

    return card


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _build_products() -> List[Product]:
    categories = ["pin", "brooch", "earrings", "bag", "shoes"]
    styles = ["minimal", "vintage", "playful", "outdoor", "craft"]
    materials = ["brass", "steel", "resin", "canvas", "leather"]
    colors = ["green", "blue", "red", "black", "white", "brown", "gold", "pink", "gray"]
    brands = ["northwind", "atelier", "fieldnote", "amberfox", "fernlab"]
    adjectives = ["botanical", "artisan", "retro", "textured", "clean", "ornate", "soft", "bold"]
    nouns = {
        "pin": ["leaf pin", "beer hop pin", "flora badge"],
        "brooch": ["pine brooch", "owl brooch", "nature brooch"],
        "earrings": ["stone earrings", "drift earrings", "flower studs"],
        "bag": ["crossbody bag", "canvas tote", "mini satchel"],
        "shoes": ["runner shoes", "city sneakers", "trail shoes"],
    }
    prices = ["budget", "mid", "premium"]

    products: List[Product] = []
    item_id = 0
    rng = random.Random(7)
    for category in categories:
        for style in styles:
            for material in materials:
                for _variant in range(2):
                    color = rng.choice(colors)
                    brand = rng.choice(brands)
                    adjective = rng.choice(adjectives)
                    noun = rng.choice(nouns[category])
                    title = f"{adjective} {color} {noun}"
                    price_bucket = rng.choice(prices)
                    products.append(
                        Product(
                            item_id=item_id,
                            title=title,
                            brand=brand,
                            category=category,
                            style=style,
                            material=material,
                            color=color,
                            price_bucket=price_bucket,
                        )
                    )
                    item_id += 1
                    if item_id >= 220:
                        return products
    return products


def _product_meta_text(product: Product) -> str:
    return (
        f"title {product.title} brand {product.brand} category {product.category} "
        f"style {product.style} material {product.material} color {product.color} price {product.price_bucket}"
    )


def _sample_user_examples(products: Sequence[Product], seed: int = 17) -> List[UserExample]:
    rng = random.Random(seed)
    examples: List[UserExample] = []
    grouped: Dict[Tuple[str, str], List[Product]] = {}
    for p in products:
        grouped.setdefault((p.category, p.style), []).append(p)

    user_pref_keys = list(grouped.keys())
    rng.shuffle(user_pref_keys)
    for pref_idx, pref in enumerate(user_pref_keys[:48]):
        candidates = grouped[pref]
        if len(candidates) < 6:
            continue
        rng.shuffle(candidates)
        history = candidates[:4]
        target = candidates[4]
        alt = candidates[5]
        preferred_material = history[0].material if pref_idx % 2 == 0 else target.material
        query_text = (
            f"looking for a {target.style} {target.category} in {target.color} "
            f"with {preferred_material} feel from {target.brand} and {target.price_bucket} pricing"
        )
        examples.append(
            UserExample(
                history_ids=[p.item_id for p in history],
                positive_id=target.item_id,
                query_text=query_text,
            )
        )
        query_text_2 = (
            f"need a {alt.category} that feels {alt.style} and {alt.material} "
            f"with {alt.color} tone and {alt.price_bucket} price"
        )
        examples.append(
            UserExample(
                history_ids=[p.item_id for p in history[:3]] + [target.item_id],
                positive_id=alt.item_id,
                query_text=query_text_2,
            )
        )
    return examples


def _build_vocab(products: Sequence[Product], examples: Sequence[UserExample]) -> Dict[str, int]:
    tokens = {"<pad>", "<unk>"}
    for p in products:
        tokens.update(_normalize(_product_meta_text(p)).split())
    for ex in examples:
        tokens.update(_normalize(ex.query_text).split())
    ordered = ["<pad>", "<unk>"] + sorted(tok for tok in tokens if tok not in {"<pad>", "<unk>"})
    return {tok: idx for idx, tok in enumerate(ordered)}


def _split_examples(examples: Sequence[UserExample]) -> Tuple[List[UserExample], List[UserExample], List[UserExample]]:
    shuffled = list(examples)
    rng = random.Random(20261009)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return list(shuffled[:train_end]), list(shuffled[train_end:val_end]), list(shuffled[val_end:])


def build_prepared_data(
    seed: int = 17,
    canvas_size: int = 224,
    image_height: int = 128,
    max_query_len: int = 24,
    max_meta_len: int = 40,
) -> PreparedData:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    products = _build_products()
    examples = _sample_user_examples(products, seed=seed)
    vocab = _build_vocab(products, examples)
    pad_id, unk_id = vocab["<pad>"], vocab["<unk>"]
    tokenizer = SimpleTokenizer(vocab, pad_id=pad_id, unk_id=unk_id)

    cards: List[torch.Tensor] = []
    images: List[torch.Tensor] = []
    meta_ids: List[List[int]] = []
    meta_masks: List[List[int]] = []
    for product in products:
        base = _make_base_image(product, size=image_height)
        card = render_product_card(product, canvas_size=canvas_size, image_height=image_height)
        cards.append(pil_to_tensor(card))
        images.append(pil_to_tensor(base))
        ids, mask = tokenizer.encode(_product_meta_text(product), max_meta_len)
        meta_ids.append(ids)
        meta_masks.append(mask)

    train_examples, val_examples, test_examples = _split_examples(examples)

    return PreparedData(
        products=products,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        vocab=vocab,
        pad_id=pad_id,
        unk_id=unk_id,
        max_query_len=max_query_len,
        max_meta_len=max_meta_len,
        card_tensors=torch.stack(cards),
        image_tensors=torch.stack(images),
        meta_tokens=torch.tensor(meta_ids, dtype=torch.long),
        meta_masks=torch.tensor(meta_masks, dtype=torch.float32),
    )


def encode_queries(
    examples: Sequence[UserExample],
    vocab: Dict[str, int],
    max_query_len: int,
    pad_id: int,
    unk_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = SimpleTokenizer(vocab=vocab, pad_id=pad_id, unk_id=unk_id)
    ids, masks = [], []
    for ex in examples:
        tok_ids, tok_mask = tokenizer.encode(ex.query_text, max_query_len)
        ids.append(tok_ids)
        masks.append(tok_mask)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.float32)


def stack_histories(examples: Sequence[UserExample]) -> torch.Tensor:
    return torch.tensor([ex.history_ids for ex in examples], dtype=torch.long)


def positive_ids(examples: Sequence[UserExample]) -> torch.Tensor:
    return torch.tensor([ex.positive_id for ex in examples], dtype=torch.long)


def sample_negative_ids(
    examples: Sequence[UserExample],
    num_items: int,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    negatives = []
    for ex in examples:
        blocked = set(ex.history_ids + [ex.positive_id])
        cand = rng.randrange(num_items)
        while cand in blocked:
            cand = rng.randrange(num_items)
        negatives.append(cand)
    return torch.tensor(negatives, dtype=torch.long)
