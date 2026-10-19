from dataclasses import dataclass
from typing import Dict, List
import re


FIELDS = ("title", "features", "description")


@dataclass
class Product:
    product_id: str
    category: str
    title: str
    features: str
    description: str

    def as_text(self) -> str:
        return f"{self.title} {self.features} {self.description}"

    def copy_with(self, updates: Dict[str, str]) -> "Product":
        payload = {field: getattr(self, field) for field in FIELDS}
        payload.update(updates)
        return Product(self.product_id, self.category, **payload)


@dataclass
class QueryCase:
    query: str
    target_id: str
    candidates: List[Product]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(cases: List[QueryCase]) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for case in cases:
        for token in tokenize(case.query):
            vocab.setdefault(token, len(vocab))
        for product in case.candidates:
            for token in tokenize(product.as_text()):
                vocab.setdefault(token, len(vocab))
    return vocab


def bow(text: str, vocab: Dict[str, int]):
    import torch

    x = torch.zeros(len(vocab), dtype=torch.float32)
    for token in tokenize(text):
        x[vocab.get(token, 1)] += 1.0
    return x / max(1.0, x.sum())


def toy_cases() -> List[QueryCase]:
    cases = [
        QueryCase(
            query="quiet air compressor quick refill for airbrush",
            target_id="compressor_a",
            candidates=[
                Product("compressor_a", "tools", "Rooty Tooty Airbrush Compressor", "low-noise motor, 3L tank, quick pressure recovery", "Compact compressor for airbrush painting and indoor craft work."),
                Product("compressor_b", "tools", "Garage Tire Inflator", "high PSI gauge, long hose", "Built for tires and sports balls with louder garage operation."),
                Product("compressor_c", "tools", "Mini USB Pump", "portable, low power", "Small pump for toys and occasional inflation."),
            ],
        ),
        QueryCase(
            query="aux cable for car that will not break after a month",
            target_id="aux_cable",
            candidates=[
                Product("aux_cable", "electronics", "CableCreation 6 FT Flat 3.5mm Aux Cable", "right-angle plug, TPE jacket, inner enameled wire", "Flat cable for car audio and phone connections."),
                Product("bluetooth_adapter", "electronics", "Bluetooth FM Adapter", "wireless audio, USB charging", "Streams phone audio through FM radio."),
                Product("short_aux", "electronics", "Short Aux Cord", "basic PVC cable", "A budget cord for occasional listening."),
            ],
        ),
        QueryCase(
            query="waterproof leather work boots with stitched seams",
            target_id="boots",
            candidates=[
                Product("boots", "apparel", "Full-Grain Leather Work Boots", "triple-stitched seams, EVA outsole, water-resistant finish", "Durable boots for shop floors and outdoor work."),
                Product("sneakers", "apparel", "Lightweight Walking Sneakers", "mesh upper, foam sole", "Comfortable shoes for daily walking."),
                Product("rain_boots", "apparel", "PVC Rain Boots", "waterproof shell, pull tabs", "Simple boots for wet weather."),
            ],
        ),
        QueryCase(
            query="coffee grinder with consistent coarse grind for french press",
            target_id="grinder",
            candidates=[
                Product("grinder", "kitchen", "Burr Coffee Grinder", "40 grind settings, stainless burrs, coarse mode", "Consistent grind size for espresso, pour-over, and French press."),
                Product("blade_grinder", "kitchen", "Blade Spice Grinder", "one-touch blade, compact cup", "Fast chopping for spices and coffee."),
                Product("manual_mill", "kitchen", "Manual Coffee Mill", "ceramic burr, hand crank", "Portable grinder for travel."),
            ],
        ),
    ]
    return cases
