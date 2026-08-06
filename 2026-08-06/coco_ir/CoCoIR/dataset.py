import torch
from torch.utils.data import Dataset


class ToyCoCoIRDataset(Dataset):
    def __init__(self, num_dialogues=384, turns=4, image_dim=32, text_dim=32, seed=7):
        generator = torch.Generator().manual_seed(seed)
        self.source_images = torch.randn(num_dialogues, image_dim, generator=generator)
        self.instructions = torch.randn(num_dialogues, turns, text_dim, generator=generator)
        transform = torch.randn(text_dim, image_dim, generator=generator) / text_dim ** 0.5
        states = []
        current_image = self.source_images.clone()
        for turn_index in range(turns):
            current_image = torch.tanh(current_image + self.instructions[:, turn_index] @ transform + 0.05 * torch.randn(num_dialogues, image_dim, generator=generator))
            states.append(current_image.clone())
        self.targets = torch.stack(states, dim=1)
        self.candidates = torch.randn(num_dialogues, turns, 16, image_dim, generator=generator)
        self.candidates[:, :, 0, :] = self.targets
        permutation = torch.argsort(torch.randn(num_dialogues, turns, 16, generator=generator), dim=-1)
        self.candidates = torch.gather(self.candidates, 2, permutation.unsqueeze(-1).expand(-1, -1, -1, image_dim))
        self.labels = torch.argmin((self.candidates - self.targets.unsqueeze(2)).pow(2).sum(dim=-1), dim=-1)

    def __len__(self):
        return self.source_images.size(0)

    def __getitem__(self, index):
        return {
            "source_image": self.source_images[index],
            "instructions": self.instructions[index],
            "candidates": self.candidates[index],
            "labels": self.labels[index],
        }
