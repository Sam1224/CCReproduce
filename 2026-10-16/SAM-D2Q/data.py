from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image, ImageDraw


STOPWORDS = {"for", "the", "with", "and", "new", "women", "men"}


@dataclass(frozen=True)
class Product:
    product_id: int
    title: str
    category: str
    color: str
    style: str
    material: str
    hidden_visual_terms: Tuple[str, ...]
    commerce_weight: float
    gold_queries: Tuple[str, ...]


@dataclass(frozen=True)
class TrainingExample:
    product_id: int
    title: str
    masked_title: str
    target_terms: Tuple[str, ...]
    hidden_visual_terms: Tuple[str, ...]
    commerce_weight: float


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    target_product_id: int
    quality: float


@dataclass
class ToyDataset:
    products: List[Product]
    train_examples: List[TrainingExample]
    val_examples: List[TrainingExample]
    test_examples: List[TrainingExample]
    eval_queries: List[RetrievalQuery]
    vocab: Dict[str, int]
    id_to_token: List[str]
    image_tensors: torch.Tensor
    title_tokens: torch.Tensor
    title_masks: torch.Tensor
    masked_title_tokens: torch.Tensor
    masked_title_masks: torch.Tensor
    target_vectors: torch.Tensor
    visual_term_vectors: torch.Tensor


COLOR_TO_RGB = {
    "red": (220, 60, 60),
    "blue": (60, 120, 220),
    "green": (70, 170, 90),
    "black": (40, 40, 40),
    "pink": (220, 120, 180),
    "gold": (220, 180, 60),
}


STYLE_MARKS = {
    "floral": "flower",
    "casual": "dot",
    "sport": "stripe",
    "minimal": "line",
}


def tokenize(text: str) -> List[str]:
    return [tok.strip().lower() for tok in text.split() if tok.strip()]


def mask_title_terms(title: str, hidden_terms: Sequence[str]) -> str:
    keep = [tok for tok in tokenize(title) if tok not in set(hidden_terms)]
    return " ".join(keep)


def build_products() -> List[Product]:
    specs = [
        ("dress", "red", "floral", "cotton"),
        ("dress", "blue", "casual", "linen"),
        ("shirt", "green", "casual", "cotton"),
        ("shirt", "black", "minimal", "linen"),
        ("sneaker", "red", "sport", "mesh"),
        ("sneaker", "blue", "sport", "mesh"),
        ("bag", "black", "minimal", "leather"),
        ("bag", "pink", "floral", "canvas"),
        ("lamp", "gold", "minimal", "metal"),
        ("lamp", "black", "minimal", "metal"),
        ("chair", "green", "minimal", "wood"),
        ("chair", "blue", "casual", "wood"),
    ]
    products: List[Product] = []
    for idx, (category, color, style, material) in enumerate(specs):
        if category in {"dress", "shirt"}:
            hidden = (color, style)
            title = f"{category} summer wear"
        elif category == "sneaker":
            hidden = (color, style)
            title = f"{category} running shoe"
        elif category == "bag":
            hidden = (color, style, material)
            title = f"{category} everyday carry"
        elif category == "lamp":
            hidden = (color, material)
            title = f"{category} home decor"
        else:
            hidden = (color, material)
            title = f"{category} living room"
        gold_queries = (
            f"{color} {style} {category}",
            f"{material} {color} {category}",
            f"{color} {category}",
        )
        commerce_weight = 1.0 + (idx % 4) * 0.25
        products.append(
            Product(
                product_id=idx,
                title=title,
                category=category,
                color=color,
                style=style,
                material=material,
                hidden_visual_terms=tuple(hidden),
                commerce_weight=commerce_weight,
                gold_queries=gold_queries,
            )
        )
    return products


def render_product_image(product: Product, size: int = 64) -> torch.Tensor:
    image = Image.new("RGB", (size, size), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    color = COLOR_TO_RGB[product.color]
    margin = 10
    draw.rounded_rectangle((margin, margin, size - margin, size - margin), radius=10, fill=color)
    mark = STYLE_MARKS[product.style]
    if mark == "flower":
        draw.ellipse((18, 18, 30, 30), fill=(255, 235, 235))
        draw.ellipse((34, 18, 46, 30), fill=(255, 235, 235))
        draw.ellipse((26, 28, 38, 40), fill=(255, 235, 235))
    elif mark == "stripe":
        for offset in range(14, size - 14, 8):
            draw.line((offset, 16, offset, size - 16), fill=(255, 255, 255), width=3)
    elif mark == "line":
        draw.line((18, size // 2, size - 18, size // 2), fill=(255, 255, 255), width=4)
    else:
        for row in range(18, size - 18, 14):
            for col in range(18, size - 18, 14):
                draw.ellipse((col, row, col + 4, row + 4), fill=(255, 255, 255))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(size, size, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    return tensor


def collect_vocab(products: Sequence[Product]) -> List[str]:
    terms = {"<pad>", "<unk>"}
    for product in products:
        terms.update(tokenize(product.title))
        terms.update(tokenize(mask_title_terms(product.title, product.hidden_visual_terms)))
        for query in product.gold_queries:
            terms.update(tok for tok in tokenize(query) if tok not in STOPWORDS)
        terms.add(product.category)
        terms.add(product.color)
        terms.add(product.style)
        terms.add(product.material)
    return sorted(terms)


def encode_texts(
    texts: Sequence[str],
    vocab: Dict[str, int],
    max_len: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pad_id = vocab["<pad>"]
    unk_id = vocab["<unk>"]
    tokens = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
    masks = torch.zeros((len(texts), max_len), dtype=torch.float32)
    for row, text in enumerate(texts):
        ids = [vocab.get(tok, unk_id) for tok in tokenize(text)[:max_len]]
        if not ids:
            continue
        tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        masks[row, : len(ids)] = 1.0
    return tokens, masks


def build_target_vector(product: Product, vocab: Dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    for query in product.gold_queries:
        for tok in tokenize(query):
            if tok in STOPWORDS:
                continue
            vector[vocab[tok]] = 1.0
    return vector


def build_visual_vector(product: Product, vocab: Dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    for tok in product.hidden_visual_terms:
        vector[vocab[tok]] = 1.0
    return vector


def split_examples(examples: Sequence[TrainingExample]) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    train = list(examples[:8])
    val = list(examples[8:10])
    test = list(examples[10:])
    return train, val, test


def build_training_examples(products: Sequence[Product]) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    for product in products:
        target_terms = sorted(
            {
                tok
                for query in product.gold_queries
                for tok in tokenize(query)
                if tok not in STOPWORDS
            }
        )
        examples.append(
            TrainingExample(
                product_id=product.product_id,
                title=product.title,
                masked_title=mask_title_terms(product.title, product.hidden_visual_terms),
                target_terms=tuple(target_terms),
                hidden_visual_terms=product.hidden_visual_terms,
                commerce_weight=product.commerce_weight,
            )
        )
    return examples


def build_eval_queries(products: Sequence[Product]) -> List[RetrievalQuery]:
    queries: List[RetrievalQuery] = []
    for product in products:
        for idx, query in enumerate(product.gold_queries):
            queries.append(
                RetrievalQuery(
                    query=query,
                    target_product_id=product.product_id,
                    quality=product.commerce_weight + 0.1 * idx,
                )
            )
    return queries


def build_toy_dataset(seed: int = 17) -> ToyDataset:
    random.seed(seed)
    products = build_products()
    vocab_list = collect_vocab(products)
    vocab = {tok: idx for idx, tok in enumerate(vocab_list)}

    examples = build_training_examples(products)
    train_examples, val_examples, test_examples = split_examples(examples)

    image_tensors = torch.stack([render_product_image(product) for product in products], dim=0)
    title_tokens, title_masks = encode_texts([product.title for product in products], vocab)
    masked_title_tokens, masked_title_masks = encode_texts([mask_title_terms(product.title, product.hidden_visual_terms) for product in products], vocab)
    target_vectors = torch.stack([build_target_vector(product, vocab) for product in products], dim=0)
    visual_term_vectors = torch.stack([build_visual_vector(product, vocab) for product in products], dim=0)

    return ToyDataset(
        products=products,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        eval_queries=build_eval_queries(products),
        vocab=vocab,
        id_to_token=vocab_list,
        image_tensors=image_tensors,
        title_tokens=title_tokens,
        title_masks=title_masks,
        masked_title_tokens=masked_title_tokens,
        masked_title_masks=masked_title_masks,
        target_vectors=target_vectors,
        visual_term_vectors=visual_term_vectors,
    )


def retrieve_with_lexical_index(query: str, doc_terms: Dict[int, Sequence[str]], topk: int = 5) -> List[int]:
    query_terms = [tok for tok in tokenize(query) if tok not in STOPWORDS]
    scored: List[Tuple[float, int]] = []
    for product_id, terms in doc_terms.items():
        doc_set = set(terms)
        overlap = len([tok for tok in query_terms if tok in doc_set])
        exact_bonus = 0.5 if all(tok in doc_set for tok in query_terms) else 0.0
        score = overlap + exact_bonus
        scored.append((score, product_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [product_id for _, product_id in scored[:topk]]


def build_title_only_index(products: Sequence[Product]) -> Dict[int, List[str]]:
    return {product.product_id: tokenize(product.title) for product in products}


def build_expanded_index(
    products: Sequence[Product],
    generated_terms: Dict[int, Sequence[str]],
) -> Dict[int, List[str]]:
    index: Dict[int, List[str]] = {}
    for product in products:
        merged = list(tokenize(product.title))
        merged.extend(tok for tok in generated_terms.get(product.product_id, []) if tok not in merged)
        index[product.product_id] = merged
    return index


def evaluate_index(
    queries: Sequence[RetrievalQuery],
    products: Sequence[Product],
    doc_terms: Dict[int, Sequence[str]],
    topk: int = 5,
) -> Dict[str, float]:
    product_map = {product.product_id: product for product in products}
    relevant_count = 0.0
    total_quality = 0.0
    zero_result = 0.0
    recall = 0.0
    for query in queries:
        ranked = retrieve_with_lexical_index(query.query, doc_terms, topk=topk)
        if not ranked:
            zero_result += 1.0
            continue
        hit = query.target_product_id in ranked
        recall += 1.0 if hit else 0.0
        if hit:
            relevant_count += 1.0
            total_quality += query.quality
        top_score = len(set(tokenize(query.query)).intersection(set(doc_terms[ranked[0]])))
        if top_score == 0:
            zero_result += 1.0
    denom = max(len(queries), 1)
    topk_name = f"@{topk}"
    return {
        f"Recall{topk_name}": recall / denom,
        f"Relevant{topk_name}": relevant_count / denom,
        f"TotalQuality{topk_name}": total_quality,
        "ZeroResultRate": zero_result / denom,
    }
