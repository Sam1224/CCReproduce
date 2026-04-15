"""Toy reproduction of ORCA (Online Reasoning Calibration).

This implementation is a lightweight, runnable approximation of the paper:
"Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning"
(arXiv:2604.01170).

It reproduces the core ideas in a synthetic setting:
- a confidence probe over step-wise "reasoning" features
- per-instance test-time training (TTT) of fast weights
- conformal-style threshold calibration for early stopping under risk control
"""

from .models import ConfidenceProbe
from .data import make_splits, SyntheticReasoningDataset
