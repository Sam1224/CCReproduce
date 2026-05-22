"""Toy e-commerce I2I retrieval dataset.

Items have: image (RGB tensor), structured metadata text (category/brand/attributes),
and a noise_mask flag indicating whether the image has promotional overlays.
Positive pairs share the same product_id; negatives are sampled randomly.
"""
import random
import torch
from torch.utils.data import Dataset


class ECommerceDataset(Dataset):
    """Synthetic I2I retrieval dataset.

    Each item: {"image": (C,H,W), "text_tokens": (L,), "product_id": int, "noisy": bool}
    Pair (query, positive) shares product_id; negatives are distinct product_ids.
    """

    def __init__(
        self,
        num_products: int = 200,
        images_per_product: int = 3,
        image_size: int = 32,
        text_seq_len: int = 16,
        vocab_size: int = 512,
        noise_ratio: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        rng = random.Random(seed)
        torch.manual_seed(seed)

        self.items = []
        for pid in range(num_products):
            # Product-specific visual prototype (embeddings will cluster around this)
            proto = torch.randn(3, image_size, image_size) * 0.5
            for _ in range(images_per_product):
                noisy = rng.random() < noise_ratio
                img = proto + torch.randn_like(proto) * 0.2
                if noisy:
                    # Simulate promotional overlay: corrupt a random 40% patch
                    h, w = image_size, image_size
                    ph, pw = h // 2, w // 2
                    r = rng.randint(0, h - ph)
                    c = rng.randint(0, w - pw)
                    img[:, r : r + ph, c : c + pw] = torch.rand(3, ph, pw)
                img = img.clamp(-1, 1)

                # Metadata text: product_id-specific tokens
                base_tokens = torch.randint(1, vocab_size, (text_seq_len,))
                # Shift first tokens to encode product identity
                base_tokens[0] = pid % (vocab_size - 1) + 1
                base_tokens[1] = (pid // 10) % (vocab_size - 1) + 1

                self.items.append(
                    {
                        "image": img,
                        "text_tokens": base_tokens,
                        "product_id": pid,
                        "noisy": noisy,
                    }
                )

        self.product_ids = list(range(num_products))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        query = self.items[idx]
        pid = query["product_id"]

        # Positive: different image, same product
        same_pid_indices = [
            i for i, it in enumerate(self.items) if it["product_id"] == pid and i != idx
        ]
        pos_idx = random.choice(same_pid_indices) if same_pid_indices else idx
        positive = self.items[pos_idx]

        # Negative: random item with different product_id
        neg_idx = random.choice(
            [i for i, it in enumerate(self.items) if it["product_id"] != pid]
        )
        negative = self.items[neg_idx]

        return {
            "q_image": query["image"],
            "q_text": query["text_tokens"],
            "pos_image": positive["image"],
            "pos_text": positive["text_tokens"],
            "neg_image": negative["image"],
            "neg_text": negative["text_tokens"],
            "q_noisy": torch.tensor(float(query["noisy"])),
        }
