"""
UNIVID toy data generator.

Produces a synthetic dataset of (video_features, policy_id, caption_tokens, violation_label)
with two label types:
  - "human"  : gold-standard annotation (clean, rare)
  - "synthetic": LLM-generated annotation (noisier, abundant)

The dataset simulates 6 content policies common in e-commerce/short-video platforms.
"""

import argparse
import os
import random
import torch
from torch.utils.data import Dataset

POLICIES = [
    "body_shaping_claims",       # unsubstantiated slimming / toning claims
    "health_supplement_claims",  # medical or cure claims without evidence
    "counterfeit_goods",         # fake brand / luxury knockoff
    "explicit_content",          # adult / NSFW
    "hate_speech",               # discriminatory language or imagery
    "misleading_pricing",        # bait-and-switch, hidden fees
]

# Vocabulary for toy caption generation (word-id mapping)
# Real UNIVID uses a LLM tokenizer; here we use a tiny fixed vocab.
CAPTION_VOCAB = (
    ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    + [f"word_{i}" for i in range(200)]
    + ["VIOLATION", "NO_VIOLATION", "POLICY", "VIDEO", "CONTENT",
       "CLAIM", "PRODUCT", "MISLEAD", "HARM", "SAFE", "CONCERN",
       "EXPLICIT", "FAKE", "HEALTH", "SLIM", "BRAND", "PRICE"]
)
VOCAB_SIZE = len(CAPTION_VOCAB)
W2I = {w: i for i, w in enumerate(CAPTION_VOCAB)}


def _rand_caption(policy: str, is_violation: bool, length: int = 12) -> list[int]:
    """Generate a toy caption as a list of token ids."""
    tokens = [W2I["<BOS>"]]
    key_words = {
        "body_shaping_claims": ["SLIM", "CLAIM", "HARM"],
        "health_supplement_claims": ["HEALTH", "CLAIM", "CONCERN"],
        "counterfeit_goods": ["FAKE", "BRAND", "PRODUCT"],
        "explicit_content": ["EXPLICIT", "CONTENT", "HARM"],
        "hate_speech": ["HARM", "CONTENT", "CONCERN"],
        "misleading_pricing": ["MISLEAD", "PRICE", "CONCERN"],
    }
    if is_violation:
        # Inject policy-relevant violation signals
        for kw in key_words.get(policy, []):
            tokens.append(W2I.get(kw, W2I["<UNK>"]))
        tokens.append(W2I["VIOLATION"])
    else:
        tokens.append(W2I["SAFE"])
        tokens.append(W2I["CONTENT"])
    # Pad with generic words
    while len(tokens) < length - 1:
        tokens.append(W2I.get(f"word_{random.randint(0, 50)}", W2I["<UNK>"]))
    tokens.append(W2I["<EOS>"])
    return tokens[:length]


class UNIVIDDataset(Dataset):
    """
    Each sample:
        video_feat  : (num_frames, frame_dim) visual features
        policy_id   : int  (index into POLICIES)
        caption_ids : (seq_len,) token ids
        violation   : int  0/1
        label_type  : str  "human" | "synthetic"
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "video_feat": s["video_feat"],
            "policy_id": torch.tensor(s["policy_id"], dtype=torch.long),
            "caption_ids": torch.tensor(s["caption_ids"], dtype=torch.long),
            "violation": torch.tensor(s["violation"], dtype=torch.float),
            "label_weight": torch.tensor(s["label_weight"], dtype=torch.float),
        }


def generate_dataset(
    n_videos: int = 500,
    num_frames: int = 8,
    frame_dim: int = 64,
    seq_len: int = 12,
    human_ratio: float = 0.15,
    violation_rate: float = 0.35,
) -> UNIVIDDataset:
    """
    Args:
        n_videos     : total number of (video, policy) samples
        num_frames   : frames per video clip
        frame_dim    : visual feature dimension per frame
        seq_len      : caption token length
        human_ratio  : fraction of samples with high-quality human labels
        violation_rate: fraction of samples that are violations
    """
    samples = []
    for _ in range(n_videos):
        policy_id = random.randint(0, len(POLICIES) - 1)
        policy = POLICIES[policy_id]
        is_violation = random.random() < violation_rate
        is_human = random.random() < human_ratio

        # Visual features: random unit vectors simulating extracted frame embeddings
        video_feat = torch.randn(num_frames, frame_dim)
        if is_violation:
            # Add a synthetic "violation signal" in the first frame channel
            video_feat[0, policy_id] += 3.0

        caption_ids = _rand_caption(policy, is_violation, length=seq_len)
        # Synthetic labels are sometimes incorrect (10% noise)
        noisy_violation = is_violation
        if not is_human and random.random() < 0.10:
            noisy_violation = not is_violation

        # Human labels get higher weight in the loss (see UNIVID §3.3 training recipe)
        label_weight = 2.0 if is_human else 1.0

        samples.append(
            {
                "video_feat": video_feat,
                "policy_id": policy_id,
                "caption_ids": caption_ids,
                "violation": int(noisy_violation),
                "label_weight": label_weight,
                "label_type": "human" if is_human else "synthetic",
            }
        )
    return UNIVIDDataset(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-videos", type=int, default=500)
    parser.add_argument("--out", type=str, default="data/univid_toy.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    dataset = generate_dataset(n_videos=args.n_videos)
    torch.save(dataset.samples, args.out)
    n_viol = sum(s["violation"] for s in dataset.samples)
    n_human = sum(s["label_type"] == "human" for s in dataset.samples)
    print(
        f"Saved {len(dataset)} samples to {args.out} "
        f"[violations={n_viol} ({n_viol/len(dataset)*100:.1f}%), "
        f"human_labels={n_human} ({n_human/len(dataset)*100:.1f}%)]"
    )
