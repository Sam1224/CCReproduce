from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<think>": 3,
    "</think>": 4,
    "<answer>": 5,
    "</answer>": 6,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimpleVocab:
    def __init__(self) -> None:
        self.stoi = dict(SPECIAL_TOKENS)
        self.itos = {i: s for s, i in self.stoi.items()}

    def add(self, token: str) -> int:
        token = token.strip()
        if not token:
            return self.stoi["<pad>"]
        if token in self.stoi:
            return self.stoi[token]
        idx = len(self.stoi)
        self.stoi[token] = idx
        self.itos[idx] = token
        return idx

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = [t for t in text.replace("\n", " ").split(" ") if t]
        ids = [self.stoi["<bos>"]]
        for t in toks[: max(0, max_len - 2)]:
            ids.append(self.add(t.lower()))
        ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in (self.stoi["<pad>"], self.stoi["<bos>"], self.stoi["<eos>"]):
                continue
            out.append(self.itos.get(int(i), "<?>"))
        return " ".join(out)


@dataclass
class Example:
    frames: torch.Tensor  # (T, d_frame)
    patches: torch.Tensor  # (T, P, d_patch)
    asr_ids: torch.Tensor  # (L,)
    ocr_ids: torch.Tensor  # (L,)
    prompt_ids: torch.Tensor  # (L,)
    scene: int
    ad_type: int
    risky: int
    rationale_ids: torch.Tensor  # (R,)


def _mk_rationale(scene: int, ad_type: int, risky: int) -> str:
    risk_word = "risky" if risky else "safe"
    return (
        f"<think> evidence from video/text suggests scene_{scene} type_{ad_type} and the ad looks {risk_word} . </think> "
        f"<answer> scene={scene} type={ad_type} risky={risky} </answer>"
    )


class BLMGuardDataset(Dataset):
    def __init__(
        self,
        *,
        n: int = 4096,
        d_frame: int = 64,
        patches: int = 8,
        d_patch: int = 32,
        n_scene: int = 7,
        n_type: int = 5,
        seed: int = 7,
        jsonl_path: Optional[str] = None,
        max_text_len: int = 64,
        max_rationale_len: int = 96,
    ) -> None:
        super().__init__()
        seed_everything(seed)

        self.vocab = SimpleVocab()
        self.max_text_len = max_text_len
        self.max_rationale_len = max_rationale_len

        self.n_scene = n_scene
        self.n_type = n_type

        self.items: List[Example] = []

        if jsonl_path:
            self._load_jsonl(jsonl_path, d_frame=d_frame, patches=patches, d_patch=d_patch)
        else:
            for _ in range(n):
                T = int(np.random.randint(6, 18))
                frames = torch.randn(T, d_frame)
                patch = torch.randn(T, patches, d_patch)

                scene = int(np.random.randint(0, n_scene))
                ad_type = int(np.random.randint(0, n_type))
                risky = int(np.random.rand() < (0.15 + 0.1 * (scene % 3)))

                asr = f"ad speech about scene_{scene} product type_{ad_type}"
                ocr = f"banner text scene_{scene} discount"
                prompt = "moderate this ad and give explanation"
                rationale = _mk_rationale(scene, ad_type, risky)

                ex = Example(
                    frames=frames,
                    patches=patch,
                    asr_ids=torch.tensor(self.vocab.encode(asr, max_text_len), dtype=torch.long),
                    ocr_ids=torch.tensor(self.vocab.encode(ocr, max_text_len), dtype=torch.long),
                    prompt_ids=torch.tensor(self.vocab.encode(prompt, max_text_len), dtype=torch.long),
                    scene=scene,
                    ad_type=ad_type,
                    risky=risky,
                    rationale_ids=torch.tensor(self.vocab.encode(rationale, max_rationale_len), dtype=torch.long),
                )
                self.items.append(ex)

    def _load_jsonl(self, path: str, *, d_frame: int, patches: int, d_patch: int) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                frames = torch.tensor(row["frames"], dtype=torch.float32)
                if frames.ndim != 2 or frames.shape[-1] != d_frame:
                    raise ValueError("frames must be (T,d_frame)")

                # If patches not provided, synthesize from frames.
                if "patches" in row:
                    patch = torch.tensor(row["patches"], dtype=torch.float32)
                else:
                    T = frames.shape[0]
                    patch = frames[:, None, :d_patch].repeat(1, patches, 1)

                scene = int(row["scene"])
                ad_type = int(row["ad_type"])
                risky = int(row["risky"])

                asr = " ".join(row.get("asr", [])) if isinstance(row.get("asr"), list) else str(row.get("asr", ""))
                ocr = " ".join(row.get("ocr", [])) if isinstance(row.get("ocr"), list) else str(row.get("ocr", ""))
                prompt = str(row.get("prompt", "moderate this ad and give explanation"))
                rationale = str(row.get("rationale") or _mk_rationale(scene, ad_type, risky))

                self.items.append(
                    Example(
                        frames=frames,
                        patches=patch,
                        asr_ids=torch.tensor(self.vocab.encode(asr, self.max_text_len), dtype=torch.long),
                        ocr_ids=torch.tensor(self.vocab.encode(ocr, self.max_text_len), dtype=torch.long),
                        prompt_ids=torch.tensor(self.vocab.encode(prompt, self.max_text_len), dtype=torch.long),
                        scene=scene,
                        ad_type=ad_type,
                        risky=risky,
                        rationale_ids=torch.tensor(self.vocab.encode(rationale, self.max_rationale_len), dtype=torch.long),
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.items[idx]
        return {
            "frames": ex.frames,
            "patches": ex.patches,
            "asr_ids": ex.asr_ids,
            "ocr_ids": ex.ocr_ids,
            "prompt_ids": ex.prompt_ids,
            "scene": torch.tensor(ex.scene, dtype=torch.long),
            "ad_type": torch.tensor(ex.ad_type, dtype=torch.long),
            "risky": torch.tensor(ex.risky, dtype=torch.long),
            "rationale_ids": ex.rationale_ids,
        }


def pad_1d(seqs: List[torch.Tensor], pad: int) -> torch.Tensor:
    m = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), m), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pad = SPECIAL_TOKENS["<pad>"]

    # Video length may vary: pad frames/patches.
    T = max(int(b["frames"].shape[0]) for b in batch)
    d_frame = int(batch[0]["frames"].shape[1])
    P = int(batch[0]["patches"].shape[1])
    d_patch = int(batch[0]["patches"].shape[2])

    frames = torch.zeros(len(batch), T, d_frame)
    patches = torch.zeros(len(batch), T, P, d_patch)
    frame_mask = torch.zeros(len(batch), T, dtype=torch.bool)

    for i, b in enumerate(batch):
        t = int(b["frames"].shape[0])
        frames[i, :t] = b["frames"]
        patches[i, :t] = b["patches"]
        frame_mask[i, :t] = True

    return {
        "frames": frames,
        "patches": patches,
        "frame_mask": frame_mask,
        "asr_ids": pad_1d([b["asr_ids"] for b in batch], pad),
        "ocr_ids": pad_1d([b["ocr_ids"] for b in batch], pad),
        "prompt_ids": pad_1d([b["prompt_ids"] for b in batch], pad),
        "scene": torch.stack([b["scene"] for b in batch]),
        "ad_type": torch.stack([b["ad_type"] for b in batch]),
        "risky": torch.stack([b["risky"] for b in batch]),
        "rationale_ids": pad_1d([b["rationale_ids"] for b in batch], pad),
    }


def format_reward(seq: torch.Tensor) -> float:
    ids = seq.detach().cpu().tolist()
    try:
        think = ids.index(SPECIAL_TOKENS["<think>"])
        ans = ids.index(SPECIAL_TOKENS["<answer>"])
        return 1.0 if think < ans else 0.0
    except ValueError:
        return 0.0


def consistency_reward(seq: torch.Tensor, scene: int, ad_type: int, risky: int, vocab: SimpleVocab) -> float:
    text = vocab.decode(seq.detach().cpu().tolist())
    hit = 0
    hit += int(f"scene={scene}" in text)
    hit += int(f"type={ad_type}" in text)
    hit += int(f"risky={risky}" in text)
    return hit / 3.0


def correctness_reward(scene_pred: int, type_pred: int, risky_pred: int, scene_gt: int, type_gt: int, risky_gt: int) -> float:
    r = 0.0
    r += 1.0 if scene_pred == scene_gt else 0.0
    r += 1.0 if type_pred == type_gt else 0.0
    r += 1.0 if risky_pred == risky_gt else 0.0
    return r / 3.0
