import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["safe", "violence", "scam", "adult"]
FINE_LABELS = ["red_region", "edge_tool", "qr_grid", "money_word", "skin_region", "safe_scene"]

VOCAB = [
    "<pad>",
    "<unk>",
    "normal",
    "daily",
    "sports",
    "food",
    "sale",
    "click",
    "free",
    "money",
    "qr",
    "knife",
    "blood",
    "fight",
    "adult",
    "nude",
    "skin",
    "safe",
    "coupon",
    "report",
    "k1nife",
    "bl00d",
    "m0ney",
    "q r",
    "ad ult",
    "n ude",
]
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID["<pad>"]
UNK_ID = TOKEN_TO_ID["<unk>"]

ADV_MAP = {
    "knife": "k1nife",
    "blood": "bl00d",
    "money": "m0ney",
    "qr": "q r",
    "adult": "ad ult",
    "nude": "n ude",
}


@dataclass
class ToyContentSample:
    image: torch.Tensor
    text_tokens: torch.Tensor
    ocr_tokens: torch.Tensor
    adv_ocr_tokens: torch.Tensor
    label: torch.Tensor
    fine_labels: torch.Tensor


def encode_tokens(tokens: list[str], max_len: int) -> torch.Tensor:
    ids = [TOKEN_TO_ID.get(token, UNK_ID) for token in tokens[:max_len]]
    ids += [PAD_ID] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def build_vocab() -> dict[str, int]:
    return dict(TOKEN_TO_ID)


class XuanwuToyDataset(Dataset):
    """Synthetic multimodal moderation data.

    Each item has three signals that mirror a content-ecosystem moderation setting:
    a tiny image, user text, and OCR text.  Labels are coarse policy classes, while
    fine_labels mark small visual/textual evidence used for fine-grained perception.
    adv_ocr_tokens simulate OCR attacks such as leetspeak and inserted spaces.
    """

    def __init__(
        self,
        num_samples: int = 256,
        image_size: int = 32,
        max_text_len: int = 8,
        max_ocr_len: int = 8,
        seed: int = 7,
        adversarial_ratio: float = 0.6,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.max_ocr_len = max_ocr_len
        self.adversarial_ratio = adversarial_ratio
        self.samples = self._build_samples(seed)

    def _base_image(self, rng: random.Random) -> torch.Tensor:
        image = torch.randn(3, self.image_size, self.image_size) * 0.04 + 0.45
        # Add a soft content background so the CNN cannot solve the task from one pixel.
        stripe = torch.linspace(0.0, 0.12, self.image_size).view(1, 1, -1)
        if rng.random() < 0.5:
            image = image + stripe
        else:
            image = image + stripe.transpose(1, 2)
        return image.clamp(0.0, 1.0)

    def _stamp_visual_evidence(self, image: torch.Tensor, label: int) -> torch.Tensor:
        fine = torch.zeros(len(FINE_LABELS), dtype=torch.float32)
        h = w = self.image_size
        if label == 0:  # safe
            image[1, 4:12, 4:12] = 0.82
            fine[5] = 1.0
        elif label == 1:  # violence: small red patch + bright edge-like tool
            image[0, h // 2 - 3 : h // 2 + 4, w // 2 - 3 : w // 2 + 4] = 0.95
            for offset in range(14):
                image[:, 5 + offset, 20 + offset // 2] = torch.tensor([0.9, 0.9, 0.9])
            fine[0] = 1.0
            fine[1] = 1.0
        elif label == 2:  # scam: qr-like grid
            for row in range(8, 24, 4):
                image[:, row : row + 2, 8:24] = 0.05
            for col in range(8, 24, 4):
                image[:, 8:24, col : col + 2] = 0.05
            fine[2] = 1.0
            fine[3] = 1.0
        else:  # adult: skin-tone region
            image[0, 9:25, 9:25] = 0.92
            image[1, 9:25, 9:25] = 0.66
            image[2, 9:25, 9:25] = 0.55
            fine[4] = 1.0
        return fine

    def _tokens_for_label(self, label: int, rng: random.Random) -> tuple[list[str], list[str]]:
        if label == 0:
            text = ["normal", rng.choice(["daily", "sports", "food"]), "safe"]
            ocr = [rng.choice(["daily", "coupon", "report"]), "safe"]
        elif label == 1:
            text = ["fight", "report", rng.choice(["sports", "daily"])]
            ocr = ["knife", "blood", rng.choice(["report", "click"])]
        elif label == 2:
            text = ["sale", "click", "free"]
            ocr = ["qr", "money", "coupon", "click"]
        else:
            text = ["adult", "skin", rng.choice(["report", "click"])]
            ocr = ["adult", "nude", "skin"]
        rng.shuffle(text)
        rng.shuffle(ocr)
        return text, ocr

    def _attack_ocr(self, tokens: list[str], rng: random.Random) -> list[str]:
        attacked = []
        for token in tokens:
            if token in ADV_MAP and rng.random() < self.adversarial_ratio:
                attacked.append(ADV_MAP[token])
            else:
                attacked.append(token)
        # Insert benign distractors to mimic noisy industrial OCR output.
        if rng.random() < 0.35:
            attacked.insert(rng.randrange(len(attacked) + 1), rng.choice(["normal", "sale", "daily"]))
        return attacked

    def _build_samples(self, seed: int) -> list[ToyContentSample]:
        rng = random.Random(seed)
        torch_generator = torch.Generator().manual_seed(seed)
        old_state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch_generator.get_state())
        samples: list[ToyContentSample] = []
        for index in range(self.num_samples):
            label = index % len(CLASS_NAMES)
            if rng.random() < 0.25:
                label = rng.randrange(len(CLASS_NAMES))
            image = self._base_image(rng)
            fine_labels = self._stamp_visual_evidence(image, label)
            text, ocr = self._tokens_for_label(label, rng)
            adv_ocr = self._attack_ocr(ocr, rng)
            if label == 2:
                fine_labels[3] = 1.0 if "money" in ocr else fine_labels[3]
            samples.append(
                ToyContentSample(
                    image=image,
                    text_tokens=encode_tokens(text, self.max_text_len),
                    ocr_tokens=encode_tokens(ocr, self.max_ocr_len),
                    adv_ocr_tokens=encode_tokens(adv_ocr, self.max_ocr_len),
                    label=torch.tensor(label, dtype=torch.long),
                    fine_labels=fine_labels,
                )
            )
        torch.random.set_rng_state(old_state)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "image": sample.image.clone(),
            "text_tokens": sample.text_tokens.clone(),
            "ocr_tokens": sample.ocr_tokens.clone(),
            "adv_ocr_tokens": sample.adv_ocr_tokens.clone(),
            "label": sample.label.clone(),
            "fine_labels": sample.fine_labels.clone(),
        }


if __name__ == "__main__":
    dataset = XuanwuToyDataset(num_samples=4)
    item = dataset[0]
    print(item["image"].shape, item["text_tokens"], item["ocr_tokens"], item["label"])
