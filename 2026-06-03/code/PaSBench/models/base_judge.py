"""
Abstract base class for PaSBench-Video safety judges.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class StreamingPrediction:
    """Output of a judge at frame t."""
    issue_warning: bool          # Should a safety warning be issued now?
    warning_text: Optional[str]  # Content of the warning if issued
    confidence: float            # Confidence score [0, 1]


class BaseVideoSafetyJudge(ABC):
    """
    Abstract base class for streaming video safety judges.

    Implement predict_streaming() to create a concrete judge.
    """

    @abstractmethod
    def predict_streaming(
        self,
        frames: torch.Tensor,       # (t, C, H, W) - frames seen so far
        question: str,
        frame_index: int,           # current frame index t
    ) -> StreamingPrediction:
        """
        Given frames[0:t+1], decide whether to issue a safety warning.

        The model must operate under causal constraint: it cannot see future frames.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state between videos."""
        pass
