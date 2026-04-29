"""IECD[2] reproduction (inference-time decoding).

This package implements the core math of Instruction–Evidence Contrastive Dual-Stream Decoding
from: "Instruction-Evidence Contrastive Dual-Stream Decoding for Grounded Vision-Language Reasoning".

The code is written to be model-agnostic: any model adapter that can provide next-token logits for
an "instruction" prompt and an "evidence" prompt can be plugged in.
"""

from .core import IECDConfig, IECDStepResult, iecd_fuse_logits
