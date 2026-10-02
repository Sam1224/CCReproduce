import torch

DOMAINS = ["ecommerce", "medical", "math", "remote_sensing"]
VOCAB = [
    "product", "caption", "creator", "policy", "appeal", "violation", "review",
    "tumor", "scan", "cell", "diagnosis", "lesion", "equation", "proof", "number",
    "geometry", "satellite", "field", "road", "building", "river", "image", "text",
]

DOMAIN_KEYWORDS = {
    "ecommerce": ["product", "caption", "creator", "policy", "appeal", "violation", "review"],
    "medical": ["tumor", "scan", "cell", "diagnosis", "lesion"],
    "math": ["equation", "proof", "number", "geometry"],
    "remote_sensing": ["satellite", "field", "road", "building", "river"],
}


def make_toy_batch(domain: str, n: int = 64, feature_dim: int = 24):
    domain_index = DOMAINS.index(domain)
    x = torch.randn(n, feature_dim) * 0.25
    start = domain_index * 4
    x[:, start:start + 4] += 1.5
    text_hint = torch.zeros(n, feature_dim)
    text_hint[:, 16 + domain_index] = 1.0
    multimodal = x + text_hint
    y = torch.full((n,), domain_index, dtype=torch.long)
    return multimodal, y
