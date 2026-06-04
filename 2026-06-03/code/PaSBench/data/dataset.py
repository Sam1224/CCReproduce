"""
PaSBench-Video Dataset Loader
Paper: "PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning"
arXiv: 2606.02443

Dataset: 740 videos (481 risk + 259 no-risk) across 4 domains:
- driving, healthcare, daily life, industrial production

Risk videos are annotated with:
- risk_onset_frame: first frame where risk becomes visible
- accident_boundary_frame: last frame before accident occurs
  (the "intervention window" is between these two)
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from torch.utils.data import Dataset


DOMAINS = ["driving", "healthcare", "daily_life", "industrial"]

TASK_TYPES = [
    "temporal_occurrence",
    "temporal_counting",
    "action_description",
    "temporal_reasoning",
]


@dataclass
class VideoAnnotation:
    """Annotation for a single PaSBench-Video sample."""
    video_id: str
    domain: str
    is_risk: bool
    # Frame-level boundaries (None for no-risk videos)
    risk_onset_frame: Optional[int]
    accident_boundary_frame: Optional[int]
    # Question-answer pair
    question: str
    answer: str
    task_type: str  # one of TASK_TYPES
    # Metadata
    duration_frames: int
    fps: float


@dataclass
class StreamingVideoSample:
    """
    A sample for streaming evaluation.
    The model sees frames one-by-one (causal constraint).
    """
    annotation: VideoAnnotation
    frames: torch.Tensor  # shape: (T, C, H, W)

    @property
    def intervention_window(self) -> Optional[Tuple[int, int]]:
        """Returns (risk_onset, accident_boundary) if risk video, else None."""
        if not self.annotation.is_risk:
            return None
        return (
            self.annotation.risk_onset_frame,
            self.annotation.accident_boundary_frame,
        )


class PaSBenchVideoDataset(Dataset):
    """
    PaSBench-Video dataset for proactive safety warning evaluation.

    Supports streaming evaluation: frames are provided causally
    (model cannot look ahead past current frame index).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "all",  # "all", "risk", "no_risk"
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        frame_height: int = 224,
        frame_width: int = 224,
        max_frames: int = 128,
    ):
        self.data_dir = data_dir
        self.split = split
        self.domain = domain
        self.task_type = task_type
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.max_frames = max_frames

        self.annotations = self._load_annotations()

    def _load_annotations(self) -> List[VideoAnnotation]:
        annotation_file = os.path.join(self.data_dir, "annotations.json")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file}\n"
                "Please download PaSBench-Video dataset."
            )

        with open(annotation_file) as f:
            raw = json.load(f)

        annotations = []
        for item in raw:
            ann = VideoAnnotation(
                video_id=item["video_id"],
                domain=item["domain"],
                is_risk=item["is_risk"],
                risk_onset_frame=item.get("risk_onset_frame"),
                accident_boundary_frame=item.get("accident_boundary_frame"),
                question=item["question"],
                answer=item["answer"],
                task_type=item["task_type"],
                duration_frames=item["duration_frames"],
                fps=item["fps"],
            )

            if self.split == "risk" and not ann.is_risk:
                continue
            if self.split == "no_risk" and ann.is_risk:
                continue
            if self.domain and ann.domain != self.domain:
                continue
            if self.task_type and ann.task_type != self.task_type:
                continue

            annotations.append(ann)

        return annotations

    def _load_frames(self, video_id: str) -> torch.Tensor:
        """Load video frames as tensor of shape (T, C, H, W)."""
        import cv2

        video_path = os.path.join(self.data_dir, "videos", f"{video_id}.mp4")
        cap = cv2.VideoCapture(video_path)

        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB, resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frames.append(frame)

        cap.release()

        if not frames:
            # Return black frames as fallback
            frames = [
                np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            ]

        frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        return frames_tensor

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> StreamingVideoSample:
        ann = self.annotations[idx]
        frames = self._load_frames(ann.video_id)
        return StreamingVideoSample(annotation=ann, frames=frames)


class ToyPaSBenchDataset(Dataset):
    """
    Toy dataset for testing without real videos.
    Generates synthetic StreamingVideoSample objects.

    Statistics match paper:
    - 481 risk videos, 259 no-risk videos
    - 4 domains, 4 task types
    """

    NUM_RISK = 481
    NUM_NO_RISK = 259
    FRAME_H = 224
    FRAME_W = 224
    NUM_FRAMES = 32  # reduced for toy
    FPS = 25.0

    def __init__(self, split: str = "all", seed: int = 42):
        self.split = split
        rng = np.random.RandomState(seed)

        self.samples = []
        for i in range(self.NUM_RISK + self.NUM_NO_RISK):
            is_risk = i < self.NUM_RISK
            if split == "risk" and not is_risk:
                continue
            if split == "no_risk" and is_risk:
                continue

            domain = DOMAINS[i % len(DOMAINS)]
            task = TASK_TYPES[i % len(TASK_TYPES)]
            onset = rng.randint(5, 20) if is_risk else None
            boundary = (onset + rng.randint(3, 10)) if is_risk else None
            boundary = min(boundary, self.NUM_FRAMES - 1) if boundary else None

            ann = VideoAnnotation(
                video_id=f"toy_{i:04d}",
                domain=domain,
                is_risk=is_risk,
                risk_onset_frame=onset,
                accident_boundary_frame=boundary,
                question=f"Is there a safety risk in this {domain} scene?",
                answer="yes" if is_risk else "no",
                task_type=task,
                duration_frames=self.NUM_FRAMES,
                fps=self.FPS,
            )
            # Random frames
            frames = torch.rand(self.NUM_FRAMES, 3, self.FRAME_H, self.FRAME_W)
            self.samples.append(StreamingVideoSample(annotation=ann, frames=frames))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StreamingVideoSample:
        return self.samples[idx]
