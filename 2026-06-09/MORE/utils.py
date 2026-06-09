from __future__ import annotations

import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, *, config: dict, model: torch.nn.Module, vocab: dict[str, int]) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({"config": config, "model_state": model.state_dict(), "vocab": vocab}, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[dict, dict, dict[str, int]]:
    payload = torch.load(path, map_location=device)
    return payload["config"], payload["model_state"], payload["vocab"]


def save_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
