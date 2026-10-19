from dataclasses import dataclass
from typing import List


@dataclass
class LongQASample:
    document: List[str]
    question_points: List[str]
    answer_sentences: List[str]
    gold_supported: List[int]


def toy_samples() -> List[LongQASample]:
    return [
        LongQASample(
            document=[
                "The workshop has 12 participants.",
                "Registration costs 50 dollars per participant.",
                "Travel costs equal half the total registration costs.",
                "The budget for registration and travel is 800 dollars.",
                "The workshop starts at 09:00.",
            ],
            question_points=["registration total", "travel total", "combined total", "start time"],
            answer_sentences=[
                "Total registration costs are 600 dollars.",
                "Travel costs are 300 dollars.",
                "Registration and travel cost 900 dollars together.",
                "The start time is 09:00.",
            ],
            gold_supported=[1, 1, 1, 1],
        ),
        LongQASample(
            document=[
                "Product A has a steel housing.",
                "Product A has a TPE coated cable.",
                "The L-shaped connector reduces bend stress.",
                "Product B is wireless.",
            ],
            question_points=["material", "failure prevention"],
            answer_sentences=[
                "Product A uses steel housing and a TPE coated cable.",
                "The connector design reduces bend stress.",
            ],
            gold_supported=[1, 1],
        ),
        LongQASample(
            document=[
                "Dataset X contains 2048 long-context samples.",
                "The median breadth is 5.",
                "Depth is low in many existing datasets.",
                "Unsupported steps are filtered during synthesis.",
            ],
            question_points=["sample count", "breadth", "depth", "filter"],
            answer_sentences=[
                "Dataset X contains 2048 samples.",
                "Its median breadth is 5 while depth is often low.",
                "Unsupported steps are accepted without filtering.",
            ],
            gold_supported=[1, 1, 0],
        ),
    ]
