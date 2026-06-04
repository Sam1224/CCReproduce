"""
PaSBench-Video Evaluator

Runs the streaming evaluation pipeline:
1. Load video frame-by-frame (causal constraint)
2. At each frame, model decides whether to issue a warning
3. Record warning frame + content
4. Compute metrics against annotations
"""

import time
from typing import List, Optional, Callable
import torch
from tqdm import tqdm

from data.dataset import StreamingVideoSample, ToyPaSBenchDataset
from evaluation.metrics import (
    WarningPrediction, EvalResult, BenchmarkMetrics, compute_metrics, print_metrics
)
from models.base_judge import BaseVideoSafetyJudge


class PaSBenchEvaluator:
    """
    Streaming evaluation pipeline for PaSBench-Video.

    At each frame t, the model receives frames[0:t+1] and must decide:
    - Is a warning needed?
    - If yes, at which frame (current frame t)?
    - What is the warning content?

    Once a warning is issued, evaluation stops for that video (first-warning protocol).
    """

    def __init__(
        self,
        judge: BaseVideoSafetyJudge,
        batch_size: int = 1,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.judge = judge
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def evaluate_single(self, sample: StreamingVideoSample) -> EvalResult:
        """Evaluate a single streaming video sample."""
        ann = sample.annotation
        frames = sample.frames  # (T, C, H, W)
        T = frames.shape[0]

        warning_frame = None
        warning_text = None
        content_correct = False

        # Streaming evaluation: present frames causally
        for t in range(T):
            current_frames = frames[: t + 1]  # (t+1, C, H, W)
            prediction = self.judge.predict_streaming(
                frames=current_frames.to(self.device),
                question=ann.question,
                frame_index=t,
            )

            if prediction.issue_warning:
                warning_frame = t
                warning_text = prediction.warning_text
                # Content correctness: simple keyword check for toy data
                content_correct = self._check_content(
                    ann.answer, prediction.warning_text
                )
                break  # First-warning protocol

        return EvalResult(
            video_id=ann.video_id,
            is_risk_gt=ann.is_risk,
            is_risk_pred=(warning_frame is not None),
            warning_frame=warning_frame,
            risk_onset_frame=ann.risk_onset_frame,
            accident_boundary_frame=ann.accident_boundary_frame,
            content_correct=content_correct,
        )

    def evaluate_dataset(
        self,
        dataset,
        max_samples: Optional[int] = None,
    ) -> BenchmarkMetrics:
        """Run full benchmark evaluation."""
        results = []
        domain_labels = []
        task_labels = []

        n = min(len(dataset), max_samples) if max_samples else len(dataset)
        iterator = range(n)
        if self.verbose:
            iterator = tqdm(iterator, desc="Evaluating PaSBench-Video")

        for i in iterator:
            sample = dataset[i]
            result = self.evaluate_single(sample)
            results.append(result)
            domain_labels.append(sample.annotation.domain)
            task_labels.append(sample.annotation.task_type)

        metrics = compute_metrics(results, domain_labels, task_labels)
        if self.verbose:
            print_metrics(metrics)
        return metrics

    @staticmethod
    def _check_content(gt_answer: str, predicted_text: Optional[str]) -> bool:
        """Simple content correctness check (keyword overlap)."""
        if predicted_text is None:
            return False
        gt_words = set(gt_answer.lower().split())
        pred_words = set(predicted_text.lower().split())
        overlap = gt_words & pred_words
        return len(overlap) / max(len(gt_words), 1) >= 0.3
