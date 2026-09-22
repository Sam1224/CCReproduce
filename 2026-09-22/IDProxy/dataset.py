import torch
from torch.utils.data import Dataset


class ToyIDProxyDataset(Dataset):
    def __init__(self, size: int = 2048, users: int = 1024, items: int = 4096, vocab_size: int = 4096, seed: int = 11):
        generator = torch.Generator().manual_seed(seed)
        self.user_id = torch.randint(0, users, (size,), generator=generator)
        self.item_id = torch.randint(0, items, (size,), generator=generator)
        self.image = torch.randn(size, 3, 64, 64, generator=generator)
        self.tokens = torch.randint(1, vocab_size, (size, 16), generator=generator)
        self.is_cold = torch.rand(size, generator=generator) < 0.35
        user_signal = (self.user_id.float() % 17) / 17.0
        item_signal = (self.item_id.float() % 19) / 19.0
        content_signal = self.tokens.float().mean(dim=1) / vocab_size
        probability = torch.sigmoid(2.2 * user_signal + 1.6 * item_signal + 1.8 * content_signal - 2.0)
        self.label = torch.bernoulli(probability, generator=generator)

    def __len__(self) -> int:
        return self.label.numel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_id": self.user_id[index],
            "item_id": self.item_id[index],
            "image": self.image[index],
            "tokens": self.tokens[index],
            "is_cold": self.is_cold[index],
            "label": self.label[index],
        }
