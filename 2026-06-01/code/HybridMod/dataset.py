"""
Toy multimodal dataset for HybridMod.
In production: replace with actual livestream segments (text transcript,
audio spectrogram features, video frame embeddings).
"""
import torch
from torch.utils.data import Dataset, DataLoader


class ToyLivestreamDataset(Dataset):
    """
    Simulates a multimodal livestream moderation dataset.
    Each sample = one ~30-second livestream segment.

    Modality dimensions (typical production values):
      text_feat:   768  (BERT/RoBERTa embedding of ASR transcript)
      audio_feat:  128  (mel-spectrogram features)
      visual_feat: 2048 (ResNet / ViT frame-level features, avg-pooled)
    """

    def __init__(self, num_samples: int = 1000, num_classes: int = 10, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.text_feat = torch.randn(num_samples, 768)
        self.audio_feat = torch.randn(num_samples, 128)
        self.visual_feat = torch.randn(num_samples, 2048)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        # Binary violation flag (for precision/recall evaluation)
        self.is_violation = (self.labels > 0).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_feat": self.text_feat[idx],
            "audio_feat": self.audio_feat[idx],
            "visual_feat": self.visual_feat[idx],
            "label": self.labels[idx],
            "is_violation": self.is_violation[idx],
        }


def get_dataloaders(batch_size: int = 32, num_classes: int = 10):
    train_ds = ToyLivestreamDataset(num_samples=800, num_classes=num_classes)
    val_ds = ToyLivestreamDataset(num_samples=100, num_classes=num_classes, seed=99)
    test_ds = ToyLivestreamDataset(num_samples=100, num_classes=num_classes, seed=77)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader
