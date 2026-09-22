import torch
from torch.utils.data import Dataset


class ToyFeedDataset(Dataset):
    def __init__(self, size: int = 2048, users: int = 2048, items: int = 8192, history_len: int = 20, seed: int = 23):
        generator = torch.Generator().manual_seed(seed)
        self.user_id = torch.randint(0, users, (size,), generator=generator)
        self.history_ids = torch.randint(0, items, (size, history_len), generator=generator)
        self.target_id = torch.randint(0, items, (size,), generator=generator)
        affinity = ((self.history_ids[:, -5:].float().mean(dim=1) % 97) / 97 + (self.target_id.float() % 89) / 89) / 2
        self.click = torch.bernoulli(torch.clamp(affinity, 0.05, 0.95), generator=generator)
        self.complete = torch.bernoulli(torch.clamp(affinity * 0.8 + 0.1, 0.05, 0.95), generator=generator)
        self.duration = torch.clamp(affinity + 0.08 * torch.randn(size, generator=generator), 0.0, 1.0)

    def __len__(self) -> int:
        return self.user_id.numel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_id": self.user_id[index],
            "history_ids": self.history_ids[index],
            "target_id": self.target_id[index],
            "click": self.click[index],
            "complete": self.complete[index],
            "duration": self.duration[index],
        }
