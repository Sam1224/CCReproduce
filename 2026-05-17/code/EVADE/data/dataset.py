"""
EVADE Dataset loader with standard interface.
The real dataset requires access to the benchmark data;
this file provides the loading interface for when data is available.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class EVADEItem:
    sample_id: str
    category: str
    text: str
    image_path: Optional[str]
    label: int  # 0 = compliant, 1 = evasive violation
    task_type: str  # "single_violation" or "all_in_one"


class EVADEDataset(Dataset):
    """
    EVADE benchmark dataset loader.

    Expected data format (JSONL):
    {
        "id": "...",
        "category": "body_shaping",
        "text": "...",
        "image": "path/to/image.jpg",  # optional
        "label": 0 or 1,
        "violation_type": "..."  # optional, for analysis
    }

    Paper statistics (real dataset):
    - 2,833 text samples
    - 13,961 image samples
    - 6 categories: body shaping, height growth, health supplements,
                    medical devices, weight loss, skin care
    """

    CATEGORIES = [
        "body_shaping",
        "height_growth",
        "health_supplements",
        "medical_devices",
        "weight_loss",
        "skin_care",
    ]

    def __init__(
        self,
        data_path: str,
        task_type: str = "single_violation",
        modality: str = "text",  # "text", "image", or "multimodal"
        transform=None,
    ):
        self.data_path = Path(data_path)
        self.task_type = task_type
        self.modality = modality
        self.transform = transform
        self.items = self._load()

    def _load(self) -> list[EVADEItem]:
        items = []
        data_file = self.data_path / f"{self.task_type}_{self.modality}.jsonl"

        if not data_file.exists():
            # Fall back to toy data
            from data.toy_data import generate_toy_dataset
            toy = generate_toy_dataset(n_samples=200, task_type=self.task_type)
            return [
                EVADEItem(
                    sample_id=s.sample_id,
                    category=s.category,
                    text=s.text,
                    image_path=s.image_path,
                    label=s.label,
                    task_type=s.task_type,
                )
                for s in toy
            ]

        with open(data_file) as f:
            for line in f:
                d = json.loads(line)
                items.append(EVADEItem(
                    sample_id=d["id"],
                    category=d["category"],
                    text=d.get("text", ""),
                    image_path=d.get("image"),
                    label=d["label"],
                    task_type=self.task_type,
                ))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        result = {
            "sample_id": item.sample_id,
            "category": item.category,
            "text": item.text,
            "label": item.label,
            "task_type": item.task_type,
        }

        if item.image_path and self.modality in ("image", "multimodal"):
            try:
                image = Image.open(item.image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                result["image"] = image
            except Exception:
                result["image"] = None

        return result

    def get_stats(self) -> dict:
        from collections import Counter
        labels = [item.label for item in self.items]
        cats = [item.category for item in self.items]
        return {
            "total": len(self.items),
            "evasive": sum(labels),
            "compliant": len(labels) - sum(labels),
            "categories": dict(Counter(cats)),
        }
