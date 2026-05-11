"""
Valley3 Vision Encoder stub.
In production: replace with CLIP-ViT-L/14-336 or SigLIP.
"""

import torch
import torch.nn as nn


class VisionEncoderStub(nn.Module):
    """
    Stub vision encoder for testing.
    Production: openai/clip-vit-large-patch14-336 or google/siglip-so400m-patch14-384
    """

    def __init__(self, hidden_size: int = 1024, image_size: int = 336, patch_size: int = 14):
        super().__init__()
        self.hidden_size = hidden_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, hidden_size)
        self.num_patches = num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        B, C, H, W = pixel_values.shape
        p = 14  # patch_size
        # Patchify
        patches = pixel_values.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p * p)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)
        return self.patch_embed(patches)  # [B, num_patches, hidden_size]
