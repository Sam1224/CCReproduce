"""
VINA Toy Dataset
Simulates real/fake image and video frame distributions for AIGC detection.

Real images: smooth low-frequency content
Fake images: added high-frequency spectral peaks (simulating GAN/diffusion fingerprints)
Video frames: real/fake images + codec-like processing (blur, JPEG compression, color shift)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import io
import random


def make_real_image(size=64):
    """Smooth low-frequency image patch (real-like)."""
    arr = np.random.randn(3, size, size).astype(np.float32)
    # Smooth with large Gaussian → mostly low frequencies
    for c in range(3):
        from scipy.ndimage import gaussian_filter
        arr[c] = gaussian_filter(arr[c], sigma=4.0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def make_fake_image(size=64):
    """Real image + spectral artifact injection (fake-like)."""
    arr = make_real_image(size)
    # Inject high-frequency peaks at specific bands (GAN fingerprint simulation)
    for c in range(3):
        freq_arr = np.fft.fft2(arr[c])
        # Add energy at high-frequency grid (e.g., GAN upsampling artifacts)
        artifact = np.zeros_like(arr[c])
        for k in [size // 4, size // 3]:
            artifact[k, k] = random.uniform(0.3, 0.8)
            artifact[-k, -k] = random.uniform(0.3, 0.8)
        freq_arr += np.fft.fft2(artifact) * 0.5
        arr[c] = np.real(np.fft.ifft2(freq_arr)).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    return arr


def apply_video_codec_simulation(arr):
    """
    Simulate video codec processing on a numpy CHW float32 image.
    Applies: slight blur + JPEG compression + optional color channel shift.
    This simulates the cross-modal gap between images and video frames.
    """
    # Convert to PIL for JPEG simulation
    img = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
    # JPEG compression (simulate codec)
    buf = io.BytesIO()
    quality = random.randint(70, 95)
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    img = Image.open(buf).copy()
    # Slight Gaussian blur (simulate video processing)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    arr_out = np.array(img).astype(np.float32) / 255.0
    return arr_out.transpose(2, 0, 1)  # HWC → CHW


class ToyAIGCDataset(Dataset):
    """
    Toy dataset for VINA joint image+video training.

    Each sample has:
      - image: CHW float32 tensor (real or fake)
      - video_frame: CHW float32 tensor (same source as image, codec-processed)
      - label: 0=real, 1=fake
      - modality: 'image' or 'video'

    For VINA joint training, we return paired (image, video_frame) from same source.
    """

    def __init__(self, n_samples=2000, image_size=64, mode='train'):
        self.n_samples = n_samples
        self.image_size = image_size
        self.mode = mode
        # Pre-generate all samples for reproducibility
        random.seed(42 if mode == 'train' else 123)
        np.random.seed(42 if mode == 'train' else 123)
        self.labels = [random.randint(0, 1) for _ in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        sz = self.image_size

        if label == 0:
            img_arr = make_real_image(sz)
        else:
            img_arr = make_fake_image(sz)

        # Video frame: same source but with codec simulation (cross-modal gap)
        vid_arr = apply_video_codec_simulation(img_arr.copy())

        image = torch.from_numpy(img_arr)
        video_frame = torch.from_numpy(vid_arr)

        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        video_frame = video_frame * 2.0 - 1.0

        return image, video_frame, torch.tensor(label, dtype=torch.long)
