import torch
import torch.nn as nn


class FrequencyTeacher(nn.Module):
    def __init__(self, bins: int = 64, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bins, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )
        self.cls = nn.Linear(dim, 1)

    def forward(self, spec: torch.Tensor):
        z = self.net(spec)
        z = nn.functional.normalize(z, dim=-1)
        logit = self.cls(z).squeeze(-1)
        return z, logit


class SpatialDetector(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, dim)
        self.cls = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).view(x.shape[0], -1)
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=-1)
        logit = self.cls(z).squeeze(-1)
        return z, logit


def distill_loss(student_z: torch.Tensor, teacher_z: torch.Tensor):
    # cosine distance
    return 1.0 - (student_z * teacher_z).sum(dim=-1).mean()
