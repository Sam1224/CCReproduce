from dataclasses import dataclass
from typing import Dict, List, Tuple
import re
import torch
import torch.nn as nn

from data import LongQASample


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(samples: List[LongQASample]) -> Dict[str, int]:
    vocab = {"<unk>": 0}
    for sample in samples:
        for text in sample.document + sample.question_points + sample.answer_sentences:
            for token in tokens(text):
                vocab.setdefault(token, len(vocab))
    return vocab


def bow(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    x = torch.zeros(len(vocab))
    for token in tokens(text):
        x[vocab.get(token, 0)] += 1
    return x / max(1.0, x.sum())


class SupportScorer(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, evidence_bow: torch.Tensor, assertion_bow: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([evidence_bow, assertion_bow], dim=-1)).squeeze(-1)


@dataclass
class Step:
    evidence_ids: List[int]
    assertion_id: int
    supported: bool


@dataclass
class ReasoningGraph:
    evidence_ids: List[int]
    steps: List[Step]
    covered_points: List[bool]

    def metrics(self) -> Dict[str, float]:
        breadth = len(set(self.evidence_ids))
        # Toy depth: longest assertion chain is approximated by supported step count
        depth = max(1, sum(1 for s in self.steps if s.supported)) if self.steps else 0
        correctness = sum(1 for s in self.steps if s.supported) / max(1, len(self.steps))
        completeness = sum(self.covered_points) / max(1, len(self.covered_points))
        return {"breadth": breadth, "depth": depth, "correctness": correctness, "completeness": completeness}


class GraphAnalyzer:
    def __init__(self, scorer: SupportScorer, vocab: Dict[str, int], threshold: float = 0.5):
        self.scorer = scorer
        self.vocab = vocab
        self.threshold = threshold

    def extract_evidence(self, sample: LongQASample, assertion: str) -> List[int]:
        assertion_terms = set(tokens(assertion))
        overlaps = []
        for idx, sentence in enumerate(sample.document):
            overlap = len(assertion_terms & set(tokens(sentence)))
            if overlap:
                overlaps.append((overlap, idx))
        return [idx for _, idx in sorted(overlaps, reverse=True)[:2]]

    @torch.no_grad()
    def judge_support(self, evidence_text: str, assertion: str) -> bool:
        logit = self.scorer(bow(evidence_text, self.vocab), bow(assertion, self.vocab))
        return torch.sigmoid(logit).item() >= self.threshold

    def analyze(self, sample: LongQASample) -> ReasoningGraph:
        steps: List[Step] = []
        all_evidence = []
        for assertion_id, assertion in enumerate(sample.answer_sentences):
            evidence_ids = self.extract_evidence(sample, assertion)
            evidence_text = " ".join(sample.document[i] for i in evidence_ids)
            supported = self.judge_support(evidence_text, assertion)
            steps.append(Step(evidence_ids, assertion_id, supported))
            all_evidence.extend(evidence_ids)

        answer_text = " ".join(sample.answer_sentences).lower()
        covered = [any(t in answer_text for t in tokens(point)) for point in sample.question_points]
        return ReasoningGraph(all_evidence, steps, covered)


class LongForgeSynthesizer:
    def __init__(self, analyzer: GraphAnalyzer):
        self.analyzer = analyzer

    def filter_and_mix(self, samples: List[LongQASample], min_correctness=0.66, min_completeness=0.75) -> List[Tuple[LongQASample, Dict[str, float]]]:
        accepted = []
        for sample in samples:
            metrics = self.analyzer.analyze(sample).metrics()
            if metrics["correctness"] >= min_correctness and metrics["completeness"] >= min_completeness:
                accepted.append((sample, metrics))
        return sorted(accepted, key=lambda item: (item[1]["breadth"], item[1]["depth"]), reverse=True)
