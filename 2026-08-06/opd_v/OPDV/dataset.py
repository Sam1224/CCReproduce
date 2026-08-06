import torch
from torch.utils.data import Dataset


class ToyVisualReasoningDataset(Dataset):
    def __init__(self, num_samples=512, image_dim=48, text_dim=32, vocab_size=64, answer_len=8, seed=23):
        generator = torch.Generator().manual_seed(seed)
        self.original_image = torch.randn(num_samples, image_dim, generator=generator)
        self.zoom_image = self.original_image + 0.25 * torch.relu(self.original_image)
        self.mask_image = 0.25 * self.original_image
        self.text = torch.randn(num_samples, text_dim, generator=generator)
        latent = self.original_image[:, :answer_len] + 0.7 * self.text[:, :answer_len]
        self.labels = torch.remainder((latent * 7).long().abs(), vocab_size)

    def __len__(self):
        return self.original_image.size(0)

    def __getitem__(self, index):
        return {
            "original_image": self.original_image[index],
            "zoom_image": self.zoom_image[index],
            "mask_image": self.mask_image[index],
            "text": self.text[index],
            "labels": self.labels[index],
        }
