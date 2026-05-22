"""Toy omni e-commerce dataset for Valley3.

Supports four modalities:
  - text       : product description / user query tokens
  - image      : RGB image tensor
  - video      : short sequence of frames (T, C, H, W)
  - audio      : log-mel spectrogram (F, T) for livestream audio

Tasks:
  - classification  : product category (image/text -> label)
  - vqa             : visual question answering (image + question -> answer)
  - moderation      : violation detection (image/text/video -> binary)
  - captioning      : generate product caption (image -> text)
  - audio_cls       : audio event classification (audio -> label)
"""
import random
import torch
from torch.utils.data import Dataset


TASK_TYPES = ["classification", "vqa", "moderation", "captioning", "audio_cls"]
NUM_CLASSES = {"classification": 10, "vqa": 20, "moderation": 2, "captioning": None,
               "audio_cls": 5}


class OmniECommerceDataset(Dataset):
    """Synthetic omni-modal e-commerce dataset."""

    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 512,
        text_len: int = 32,
        image_size: int = 32,
        num_frames: int = 4,
        audio_freq: int = 40,
        audio_time: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        rng = random.Random(seed)
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.audio_freq = audio_freq
        self.audio_time = audio_time

        self.samples = []
        for _ in range(num_samples):
            task = rng.choice(TASK_TYPES)
            num_cls = NUM_CLASSES.get(task, 2)

            sample = {
                "task": task,
                "text": torch.randint(1, vocab_size, (text_len,)),
                "image": torch.randn(3, image_size, image_size).clamp(-1, 1),
                "video": torch.randn(num_frames, 3, image_size, image_size).clamp(-1, 1),
                "audio": torch.randn(audio_freq, audio_time),
            }
            if task == "captioning":
                # Target is a short token sequence
                sample["label"] = torch.randint(1, vocab_size, (text_len // 2,))
            else:
                sample["label"] = torch.tensor(rng.randint(0, (num_cls or 2) - 1))
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
