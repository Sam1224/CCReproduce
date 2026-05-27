"""
Toy dataset interface for ClaimDiff-RL reproduction.
Compatible with COCO Captions format (image_id, image_path, reference captions).
"""
import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


TOY_SAMPLES = [
    {
        "image_id": 0,
        "reference": "A red sports car parked on a sunny street next to a tree.",
        "candidate": "A blue car on a street.",
        "image_desc": "red sports car, sunny street, tree",
    },
    {
        "image_id": 1,
        "reference": "Two children playing with a yellow ball in a green park.",
        "candidate": "Children playing in the park with a ball.",
        "image_desc": "two children, yellow ball, green park",
    },
    {
        "image_id": 2,
        "reference": "A woman in a white dress holding a bouquet of pink flowers.",
        "candidate": "A woman holding flowers wearing a colorful outfit.",
        "image_desc": "woman, white dress, pink flower bouquet",
    },
    {
        "image_id": 3,
        "reference": "A black cat sleeping on a brown wooden chair near a window.",
        "candidate": "A cat sleeping on a chair.",
        "image_desc": "black cat, brown wooden chair, window",
    },
    {
        "image_id": 4,
        "reference": "A pizza with pepperoni and mushrooms on a white plate.",
        "candidate": "A pizza with cheese and vegetables on a plate.",
        "image_desc": "pizza, pepperoni, mushrooms, white plate",
    },
]


class ToyCapDataset(Dataset):
    """
    Toy captioning dataset.  In a real setup, replace with COCO Captions loader
    that returns (PIL image, list[str] reference captions).
    Here we use a text-only proxy: image_desc serves as the grounding oracle.
    """

    def __init__(self, samples=None):
        self.samples = samples if samples is not None else TOY_SAMPLES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "image_id": s["image_id"],
            "reference": s["reference"],
            "image_desc": s["image_desc"],
        }


def collate_fn(batch):
    return {
        "image_ids": [b["image_id"] for b in batch],
        "references": [b["reference"] for b in batch],
        "image_descs": [b["image_desc"] for b in batch],
    }
