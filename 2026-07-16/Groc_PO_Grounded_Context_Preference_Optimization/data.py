import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


STAGES = {"object": 0, "context": 1, "reasoning": 2}
COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["shoe", "bag", "watch", "shirt"]
CLAIMS = ["authentic", "discounted", "handmade", "limited"]


@dataclass
class GrocToyConfig:
    vocab_size: int = 64
    image_dim: int = 8
    text_len: int = 8
    response_len: int = 6
    samples: int = 1200


TOKEN = {"pad": 0}
for word in COLORS + OBJECTS + CLAIMS + ["has", "not", "looks", "same", "wrong", "because", "visible", "missing", "safe", "risky"]:
    TOKEN.setdefault(word, len(TOKEN))


def encode(words: List[str], length: int) -> torch.Tensor:
    ids = [TOKEN.get(w, 1) for w in words[:length]]
    ids += [0] * (length - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class GroundedPreferenceDataset(Dataset):
    def __init__(self, n: int = 1200, seed: int = 7, cfg: GrocToyConfig = GrocToyConfig()):
        self.cfg = cfg
        rng = random.Random(seed)
        self.rows = [self._make_row(rng) for _ in range(n)]

    def _make_image(self, color_id: int, object_id: int, claim_id: int) -> torch.Tensor:
        image = torch.zeros(self.cfg.image_dim)
        image[color_id] = 1.0
        image[4 + object_id] = 1.0
        image += 0.05 * torch.randn_like(image)
        image[-1] = claim_id / max(len(CLAIMS) - 1, 1)
        return image

    def _make_row(self, rng: random.Random) -> Dict[str, torch.Tensor]:
        color_id = rng.randrange(len(COLORS))
        object_id = rng.randrange(len(OBJECTS))
        claim_id = rng.randrange(len(CLAIMS))
        corrupt = rng.random() < 0.5
        wrong_color = (color_id + rng.randrange(1, len(COLORS))) % len(COLORS)
        wrong_object = (object_id + rng.randrange(1, len(OBJECTS))) % len(OBJECTS)
        stage_name, stage_id = rng.choice(list(STAGES.items()))

        image = self._make_image(color_id, object_id, claim_id)
        prompt = encode(["has", COLORS[color_id], OBJECTS[object_id], CLAIMS[claim_id], "visible"], self.cfg.text_len)

        if stage_name == "object":
            chosen = [COLORS[color_id], OBJECTS[object_id], "visible", "safe"]
            rejected = [COLORS[wrong_color], OBJECTS[wrong_object], "visible", "wrong"]
        elif stage_name == "context":
            chosen = [OBJECTS[object_id], "has", CLAIMS[claim_id], "because", "visible"]
            rejected = [OBJECTS[object_id], "has", CLAIMS[(claim_id + 1) % len(CLAIMS)], "because", "missing"]
        else:
            chosen = ["safe", "because", COLORS[color_id], OBJECTS[object_id], "same"]
            rejected = ["risky" if not corrupt else "safe", "because", COLORS[wrong_color], OBJECTS[object_id], "wrong"]

        if corrupt:
            chosen, rejected = rejected, chosen
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return {
            "image": image.float(),
            "prompt": prompt,
            "stage": torch.tensor(stage_id, dtype=torch.long),
            "chosen": encode(chosen, self.cfg.response_len),
            "rejected": encode(rejected, self.cfg.response_len),
            "label": label,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


def make_loaders(batch_size: int = 64):
    train = GroundedPreferenceDataset(1200, seed=11)
    test = GroundedPreferenceDataset(300, seed=19)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)
