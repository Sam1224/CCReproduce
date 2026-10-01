from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

from data import FEATURE_DIM


@dataclass
class PromptCandidate:
    name: str
    instruction_zh: str
    feature_prior: torch.Tensor


class ToyMultimodalJudge(nn.Module):
    """Small MLLM-style scorer used by the prompt search toy pipeline.

    The paper optimizes natural-language prompts of a frozen MLLM. Here each
    prompt is represented as a feature prior that shifts a shared multimodal
    risk scorer, keeping the key mechanism testable on CPU.
    """

    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.base_head = nn.Linear(hidden_dim, 1)
        self.prompt_proj = nn.Linear(FEATURE_DIM, hidden_dim, bias=False)

    def forward(self, features: torch.Tensor, prompt_prior: torch.Tensor) -> torch.Tensor:
        if prompt_prior.ndim == 1:
            prompt_prior = prompt_prior.unsqueeze(0).expand(features.shape[0], -1)
        encoded = self.encoder(features)
        prompt_shift = self.prompt_proj(prompt_prior)
        interaction = (features * prompt_prior).sum(dim=1) / (features.shape[1] ** 0.5)
        return self.base_head(encoded + 0.35 * prompt_shift).squeeze(-1) + 0.45 * interaction


def build_prompt_candidates(device: torch.device | str = "cpu") -> List[PromptCandidate]:
    prompts = []
    specs = [
        ("accuracy_majority", "优先减少误报，只有证据很强时才判违规。", [0.7, 0.2, 0.1, 0.0, 0.0, -0.3]),
        ("text_violation", "重点检查标题、caption 与话术中的夸大、引流、违规词。", [1.2, 1.0, 0.8, 0.4, 0.0, 0.0]),
        ("visual_text_consistency", "联合检查画面、商品与文本承诺是否一致。", [0.8, 0.6, 0.4, 1.1, 0.9, 0.5]),
        ("creator_governance", "结合达人历史、互动异常和商品风险做综合排序。", [0.6, 0.5, 0.5, 0.4, 1.2, 1.0]),
        ("ranking_sensitive", "不要只追求准确率，要优先把高风险内容排在低风险内容前。", [1.6, 1.2, 0.9, 1.0, 0.3, -0.8]),
        ("balanced_audit", "在漏放和误伤之间平衡，按证据强度输出连续风险分。", [1.0, 0.8, 0.6, 0.7, 0.5, 0.2]),
    ]
    for name, instruction, prefix in specs:
        prior = torch.zeros(FEATURE_DIM, dtype=torch.float32, device=device)
        values = torch.tensor(prefix, dtype=torch.float32, device=device)
        prior[: values.numel()] = values
        prior[12:15] = values[:3] * 0.8
        prior[22:25] = values[3:6] * 0.7
        prompts.append(PromptCandidate(name=name, instruction_zh=instruction, feature_prior=prior))
    return prompts


@torch.no_grad()
def score_with_prompts(
    model: ToyMultimodalJudge,
    features: torch.Tensor,
    prompts: Iterable[PromptCandidate],
) -> torch.Tensor:
    scores = []
    for prompt in prompts:
        scores.append(model(features, prompt.feature_prior.to(features.device)))
    return torch.stack(scores, dim=1)
