"""
Valley3 Omni E-commerce Benchmark — 6 Tasks.

Paper: Valley3 evaluates on an omni e-commerce benchmark spanning 6 tasks.
This implementation provides the evaluation framework with toy data.

Tasks:
  Task 1: Product Attribute Extraction (属性抽取)
          Input: product image + title → Extract: category, brand, color, size, etc.
  Task 2: Compliance/Violation Detection (合规检测)
          Input: product listing (image + text) → Output: compliant/violation type
  Task 3: Short-Video Caption Generation (短视频字幕生成)
          Input: video frames + audio → Output: Chinese product description
  Task 4: Cross-Modal Product QA (跨模态商品问答)
          Input: product image + user question → Output: accurate answer
  Task 5: Influencer Content Quality Assessment (达人内容质量评估)
          Input: KOL video (frames + audio) → Output: quality score + issues
  Task 6: Multilingual Audio Understanding (多语言音频理解)
          Input: foreign language product audio → Output: Chinese translation/summary
"""

import torch
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class BenchmarkSample:
    """A single evaluation sample."""
    task_id: int
    task_name: str
    sample_id: str
    has_image: bool
    has_video: bool
    has_audio: bool
    question: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Results for a single task."""
    task_name: str
    num_samples: int
    correct: int
    score: float
    details: List[Dict[str, Any]] = field(default_factory=list)


class EcomBenchmark:
    """
    Omni E-commerce Benchmark with 6 tasks.
    Toy data generator that mirrors the real benchmark's interface.
    """

    TASKS = {
        1: "product_attribute_extraction",
        2: "compliance_violation_detection",
        3: "short_video_caption",
        4: "crossmodal_product_qa",
        5: "influencer_content_quality",
        6: "multilingual_audio_understanding",
    }

    # Sample ground truth patterns for toy evaluation
    VIOLATION_TYPES = [
        "COMPLIANT",
        "VIOLATION:虚假宣传",
        "VIOLATION:违禁词",
        "VIOLATION:仿冒品牌",
        "VIOLATION:虚假评价",
    ]

    QUALITY_LEVELS = ["优质", "良好", "一般", "需整改", "严重违规"]

    def __init__(self, num_samples_per_task: int = 10):
        self.num_samples_per_task = num_samples_per_task
        self.samples = self._generate_samples()

    def _generate_samples(self) -> Dict[int, List[BenchmarkSample]]:
        """Generate toy benchmark samples."""
        samples = {}
        for task_id, task_name in self.TASKS.items():
            task_samples = []
            for i in range(self.num_samples_per_task):
                sample = self._create_sample(task_id, task_name, i)
                task_samples.append(sample)
            samples[task_id] = task_samples
        return samples

    def _create_sample(self, task_id: int, task_name: str, idx: int) -> BenchmarkSample:
        """Create a toy sample for a specific task."""
        if task_id == 1:  # Attribute extraction
            return BenchmarkSample(
                task_id=task_id,
                task_name=task_name,
                sample_id=f"task1_{idx:04d}",
                has_image=True, has_video=False, has_audio=False,
                question="请提取该商品的主要属性（品类、颜色、材质）",
                ground_truth=f"品类:服装;颜色:蓝色;材质:棉",
                metadata={"product_category": "服装"},
            )
        elif task_id == 2:  # Compliance detection
            gt = random.choice(self.VIOLATION_TYPES)
            return BenchmarkSample(
                task_id=task_id,
                task_name=task_name,
                sample_id=f"task2_{idx:04d}",
                has_image=True, has_video=False, has_audio=False,
                question="该商品描述是否符合平台合规要求？如有违规，请说明类型。",
                ground_truth=gt,
                metadata={"severity": "high" if "VIOLATION" in gt else "none"},
            )
        elif task_id == 3:  # Short-video caption
            return BenchmarkSample(
                task_id=task_id,
                task_name=task_name,
                sample_id=f"task3_{idx:04d}",
                has_image=False, has_video=True, has_audio=True,
                question="请为这段短视频生成商品描述文案。",
                ground_truth=f"这款商品品质优良，功能实用，性价比高，适合日常使用。",
                metadata={"language": "zh", "video_duration": random.randint(15, 60)},
            )
        elif task_id == 4:  # Cross-modal QA
            return BenchmarkSample(
                task_id=task_id,
                task_name=task_name,
                sample_id=f"task4_{idx:04d}",
                has_image=True, has_video=False, has_audio=False,
                question=random.choice([
                    "图片中商品的颜色是什么？",
                    "这个商品适合什么年龄段使用？",
                    "商品是否包含配件？",
                ]),
                ground_truth="根据图片分析，该商品为蓝色，适合18-35岁用户。",
            )
        elif task_id == 5:  # Influencer content quality
            quality = random.choice(self.QUALITY_LEVELS)
            return BenchmarkSample(
                task_id=task_id,
                task_name=task_name,
                sample_id=f"task5_{idx:04d}",
                has_image=False, has_video=True, has_audio=True,
                question="评估该达人视频内容的质量等级，并列出主要问题。",
                ground_truth=quality,
                metadata={"kol_type": random.choice(["美妆", "服装", "3C数码", "食品"])},
            )
        else:  # Task 6: Multilingual audio
            return BenchmarkSample(
                task_id=task_id,
                task_name=task_name,
                sample_id=f"task6_{idx:04d}",
                has_image=False, has_video=False, has_audio=True,
                question="请将以下英语商品介绍翻译并总结为中文。",
                ground_truth="这是一款高质量商品，具有多项实用功能，用户评价良好。",
                metadata={"source_language": "en", "target_language": "zh"},
            )

    def evaluate_sample(
        self,
        model_output: str,
        ground_truth: str,
        task_id: int,
    ) -> Tuple[bool, float]:
        """
        Simple evaluation metrics per task.
        In production: use task-specific metrics (exact match, BLEU, F1, etc.)
        """
        if task_id == 2:  # Violation detection: exact match on label
            pred_label = "VIOLATION" if "VIOLATION" in model_output else "COMPLIANT"
            gt_label = "VIOLATION" if "VIOLATION" in ground_truth else "COMPLIANT"
            correct = pred_label == gt_label
            score = 1.0 if correct else 0.0
        elif task_id in (3, 4, 6):  # Caption/QA: token overlap (simplified BLEU)
            pred_tokens = set(model_output.split())
            gt_tokens = set(ground_truth.split())
            if not gt_tokens:
                score = 0.0
            else:
                score = len(pred_tokens & gt_tokens) / len(gt_tokens)
            correct = score > 0.5
        elif task_id == 5:  # Quality assessment: label match
            correct = any(q in model_output for q in self.QUALITY_LEVELS)
            score = 1.0 if correct else 0.0
        else:  # Attribute extraction: partial match
            pred_attrs = set(model_output.replace(":", ";").split(";"))
            gt_attrs = set(ground_truth.replace(":", ";").split(";"))
            score = len(pred_attrs & gt_attrs) / max(len(gt_attrs), 1)
            correct = score > 0.6

        return correct, score


def run_evaluation(model, benchmark: EcomBenchmark, device: str = "cpu") -> Dict[str, Any]:
    """
    Run Valley3 on the e-commerce benchmark.

    Args:
        model: Valley3Model instance
        benchmark: EcomBenchmark instance
        device: compute device

    Returns:
        Dictionary of per-task and overall results
    """
    model.eval()
    model = model.to(device)

    all_results = {}
    overall_correct = 0
    overall_total = 0

    print("\n" + "=" * 60)
    print("Valley3 Omni E-commerce Benchmark Evaluation")
    print("=" * 60)

    for task_id, task_name in EcomBenchmark.TASKS.items():
        samples = benchmark.samples[task_id]
        task_correct = 0
        task_scores = []
        sample_results = []

        for sample in samples:
            # Build toy input tensors
            batch = {}
            batch["input_ids"] = torch.randint(0, 32000, (1, 16)).to(device)

            if sample.has_image or sample.has_video:
                num_frames = 4 if sample.has_video else 1
                batch["pixel_values"] = torch.randn(1, num_frames, 3, 32, 32).to(device)

            if sample.has_audio:
                batch["mel_spectrograms"] = torch.randn(1, 80, 300).to(device)

            # Forward pass (toy: just sample from vocab)
            with torch.no_grad():
                try:
                    outputs = model(**batch)
                    # In production: decode logits to text
                    # For toy: use a simple heuristic mock response
                    model_output = _mock_decode(sample.task_id, sample.ground_truth)
                except Exception as e:
                    model_output = ""

            correct, score = benchmark.evaluate_sample(
                model_output, sample.ground_truth, task_id
            )
            task_correct += int(correct)
            task_scores.append(score)
            sample_results.append({
                "sample_id": sample.sample_id,
                "correct": correct,
                "score": score,
            })

        task_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
        result = TaskResult(
            task_name=task_name,
            num_samples=len(samples),
            correct=task_correct,
            score=task_score,
            details=sample_results,
        )
        all_results[task_name] = result
        overall_correct += task_correct
        overall_total += len(samples)

        print(f"  Task {task_id} [{task_name}]: "
              f"accuracy={task_correct}/{len(samples)} ({task_score:.2%})")

    overall_accuracy = overall_correct / max(overall_total, 1)
    print(f"\nOverall: {overall_correct}/{overall_total} ({overall_accuracy:.2%})")
    print("=" * 60)

    return {
        "task_results": all_results,
        "overall_accuracy": overall_accuracy,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
    }


def _mock_decode(task_id: int, ground_truth: str) -> str:
    """
    Toy decoder for evaluation.
    In production: use autoregressive generation from Valley3Model.
    """
    # Add slight noise to simulate imperfect model
    if random.random() > 0.3:  # 70% accuracy simulation
        return ground_truth
    else:
        return "无法判断"


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from model.valley3 import build_valley3_tiny

    print("Building Valley3 (tiny)...")
    model = build_valley3_tiny()

    print("Creating benchmark...")
    benchmark = EcomBenchmark(num_samples_per_task=5)

    print("Running evaluation...")
    results = run_evaluation(model, benchmark, device="cpu")

    print(f"\nFinal Overall Accuracy: {results['overall_accuracy']:.2%}")
