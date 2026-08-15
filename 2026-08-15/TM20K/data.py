import torch
from torch.utils.data import Dataset


class ToyEcommerceSequenceDataset(Dataset):
    def __init__(self, num_samples=1024, seq_len=256, vocab_size=4096, num_categories=64, seed=7):
        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(1, vocab_size, (num_samples, seq_len), generator=generator)
        self.categories = torch.randint(0, num_categories, (num_samples, seq_len), generator=generator)
        self.positions = torch.arange(seq_len).repeat(num_samples, 1)
        recent_signal = self.tokens[:, -16:].float().mean(dim=1)
        long_signal = self.categories.float().mean(dim=1)
        threshold = recent_signal.median() + 0.03 * long_signal.median()
        self.labels = ((recent_signal + 0.03 * long_signal) > threshold).long()

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, index):
        return {
            "tokens": self.tokens[index],
            "categories": self.categories[index],
            "positions": self.positions[index],
            "label": self.labels[index],
        }
