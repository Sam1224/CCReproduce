import torch
from torch.utils.data import Dataset


class ToyGradCIRDataset(Dataset):
    def __init__(self, size: int = 512, vocab_size: int = 2048, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        self.query_images = torch.randn(size, 3, 64, 64, generator=generator)
        self.product_images = self.query_images + 0.15 * torch.randn(size, 3, 64, 64, generator=generator)
        self.modifier_tokens = torch.randint(1, vocab_size, (size, 12), generator=generator)
        self.product_tokens = self.modifier_tokens.clone()
        noise_positions = torch.rand(size, 12, generator=generator) < 0.25
        self.product_tokens[noise_positions] = torch.randint(1, vocab_size, (int(noise_positions.sum()),), generator=generator)
        image_gap = (self.query_images - self.product_images).flatten(1).pow(2).mean(dim=1)
        text_match = (self.modifier_tokens == self.product_tokens).float().mean(dim=1)
        score = 0.55 * text_match + 0.45 * torch.exp(-image_gap)
        self.relevance = torch.clamp((score * 4).long(), 0, 3)

    def __len__(self) -> int:
        return self.relevance.numel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "query_image": self.query_images[index],
            "modifier_tokens": self.modifier_tokens[index],
            "product_image": self.product_images[index],
            "product_tokens": self.product_tokens[index],
            "relevance": self.relevance[index],
        }
