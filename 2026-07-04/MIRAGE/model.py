import torch
from torch import nn

from data import ASPECTS, CONTEXTS, PRODUCTS, VOCAB


class MetadataEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(metadata), dim=-1)


class MirageLinker(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = MetadataEncoder(input_size)
        self.classifier = nn.Sequential(nn.Linear(6, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, chunk_meta: torch.Tensor, image_meta: torch.Tensor) -> torch.Tensor:
        chunk_vec = self.encoder(chunk_meta)
        image_vec = self.encoder(image_meta)
        cosine = (chunk_vec * image_vec).sum(dim=-1, keepdim=True)
        exact_overlap = (chunk_meta * image_meta).sum(dim=-1, keepdim=True) / torch.clamp(chunk_meta.sum(dim=-1, keepdim=True), min=1.0)
        distance = torch.abs(chunk_meta - image_meta).mean(dim=-1, keepdim=True)
        product_start = len(VOCAB)
        aspect_start = product_start + len(PRODUCTS)
        context_start = aspect_start + len(ASPECTS)
        product_match = (chunk_meta[:, product_start:aspect_start] * image_meta[:, product_start:aspect_start]).sum(dim=-1, keepdim=True)
        aspect_match = (chunk_meta[:, aspect_start:context_start] * image_meta[:, aspect_start:context_start]).sum(dim=-1, keepdim=True)
        context_match = (chunk_meta[:, context_start:] * image_meta[:, context_start:]).sum(dim=-1, keepdim=True)
        features = torch.cat([cosine, exact_overlap, distance, product_match, aspect_match, context_match], dim=-1)
        return self.classifier(features).squeeze(-1)


def generate_answer(chunk_text: str, image_metadata: dict) -> str:
    product = image_metadata.get("product", "the product")
    aspect = image_metadata.get("aspect", "the relevant component")
    context = image_metadata.get("context", "the reported issue")
    return f"For {context}, inspect the {aspect} on {product}. Use the linked visual only when product and aspect metadata match the troubleshooting step."


def evaluate_alignment(chunk_metadata: dict, image_metadata: dict) -> dict:
    return {
        "attribute_alignment": float(chunk_metadata.get("product") == image_metadata.get("product")),
        "aspect_alignment": float(chunk_metadata.get("aspect") == image_metadata.get("aspect")),
        "image_groundedness": float(chunk_metadata.get("context") == image_metadata.get("context")),
    }
