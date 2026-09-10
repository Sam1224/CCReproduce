from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ONTOLOGY, PATTERNS


class MetadataEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        block = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0)
        hidden = self.encoder(self.embedding(input_ids), src_key_padding_mask=~mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(self.projection(pooled), dim=-1)


class ContrastiveMetadataTagger(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int = len(ONTOLOGY), dim: int = 128):
        super().__init__()
        self.encoder = MetadataEncoder(vocab_size=vocab_size, dim=dim)
        self.classifier = nn.Linear(dim, num_tags)
        self.tag_prototypes = nn.Parameter(torch.randn(num_tags, dim) * 0.02)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encoder(input_ids)
        logits = self.classifier(embedding)
        retrieval_scores = embedding @ F.normalize(self.tag_prototypes, dim=-1).T
        return {"embedding": embedding, "logits": logits, "retrieval_scores": retrieval_scores}

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self(batch["input_ids"])
        multilabel_loss = F.binary_cross_entropy_with_logits(outputs["logits"], batch["labels"])
        contrastive_loss = F.cross_entropy(outputs["retrieval_scores"] * 20.0, batch["primary_tag"])
        return multilabel_loss + 0.35 * contrastive_loss


class RegexTagger:
    def __init__(self):
        self.patterns = PATTERNS

    def score(self, metadata_key: str, description: str = "") -> Dict[str, float]:
        text = f"{metadata_key} {description}".lower()
        scores = {tag: 0.0 for tag in ONTOLOGY}
        for tag, terms in self.patterns.items():
            hits = sum(1 for term in terms if term.lower() in text)
            if hits:
                scores[tag] = min(1.0, 0.35 + 0.2 * hits)
        if max(scores.values()) == 0.0:
            scores["NON_SENSITIVE"] = 0.5
        return scores


class DescriptionTagger:
    def score(self, description: str) -> Dict[str, float]:
        text = description.lower()
        scores = {tag: 0.0 for tag in ONTOLOGY}
        rules = {
            "CONTENT_POLICY": ["moderation", "policy", "unsafe", "caption"],
            "CREATOR_RISK": ["creator", "governance", "risk", "penalty"],
            "PRODUCT_ATTRIBUTE": ["product", "sku", "brand", "category"],
            "PAYMENT": ["payment", "billing", "card"],
            "ADDRESS": ["address", "shipping", "location"],
            "EMAIL": ["email"],
            "PHONE": ["phone", "mobile"],
            "USER_ID": ["user", "account", "seller", "buyer", "author"],
            "DEVICE_ID": ["device", "advertising"],
        }
        for tag, terms in rules.items():
            hits = sum(1 for term in terms if term in text)
            if hits:
                scores[tag] = min(1.0, 0.4 + 0.18 * hits)
        if max(scores.values()) == 0.0:
            scores["NON_SENSITIVE"] = 0.45
        return scores


def reciprocal_rank_fusion(score_maps: Sequence[Dict[str, float]], k: int = 60) -> List[Tuple[str, float, List[str]]]:
    fused: Dict[str, float] = {tag: 0.0 for tag in ONTOLOGY}
    provenance: Dict[str, List[str]] = {tag: [] for tag in ONTOLOGY}
    for source_index, scores in enumerate(score_maps):
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (tag, score) in enumerate(ranked, start=1):
            if score <= 0:
                continue
            fused[tag] += 1.0 / (k + rank)
            provenance[tag].append(f"strategy_{source_index + 1}:{score:.3f}")
    return sorted(
        [(tag, score, provenance[tag]) for tag, score in fused.items() if score > 0],
        key=lambda item: item[1],
        reverse=True,
    )


class GlyphPipeline:
    def __init__(self, model: ContrastiveMetadataTagger, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.regex_tagger = RegexTagger()
        self.description_tagger = DescriptionTagger()

    @torch.no_grad()
    def predict(self, metadata_key: str, description: str = "", line_of_business: str = "commerce", threshold: float = 0.18):
        self.model.eval()
        text = f"{metadata_key} [SEP] {description} [LOB] {line_of_business}"
        input_ids = self.tokenizer.encode(text).unsqueeze(0)
        outputs = self.model(input_ids)
        neural_scores = torch.sigmoid(outputs["logits"])[0].tolist()
        retrieval_scores = torch.softmax(outputs["retrieval_scores"][0], dim=-1).tolist()
        neural_map = {tag: float(score) for tag, score in zip(ONTOLOGY, neural_scores)}
        retrieval_map = {tag: float(score) for tag, score in zip(ONTOLOGY, retrieval_scores)}
        fused = reciprocal_rank_fusion([
            self.description_tagger.score(description),
            self.regex_tagger.score(metadata_key, description),
            neural_map,
            retrieval_map,
        ])
        accepted = [item for item in fused if item[1] >= threshold]
        return accepted[:5] or fused[:1]
