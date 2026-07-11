import torch
from torch.utils.data import Dataset


class ToyMultimodalDocumentDataset(Dataset):
    """Toy CMDR-Bench style samples with page context and hard negatives."""

    def __init__(self, length: int = 1024, pages: int = 6, feature_dim: int = 32):
        self.length = length
        self.pages = pages
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        topic = torch.randn(self.feature_dim, generator=generator)
        page_noise = torch.randn(self.pages, self.feature_dim, generator=generator) * 0.35
        pages = topic.unsqueeze(0) + page_noise
        relevant_page = torch.tensor(index % self.pages, dtype=torch.long)
        context_page = torch.tensor((index + 1) % self.pages, dtype=torch.long)
        query = 0.65 * pages[relevant_page] + 0.35 * pages[context_page] + 0.1 * torch.randn(self.feature_dim, generator=generator)
        page_images = torch.randn(self.pages, 3, 16, 16, generator=generator) * 0.2
        page_images += pages[:, :3].view(self.pages, 3, 1, 1)
        return {"query": query, "page_text": pages, "page_images": page_images, "target": relevant_page}
