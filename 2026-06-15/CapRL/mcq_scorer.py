"""
CapRL++ — MCQ-Based Verifiable Reward

Implements the core reward function: a vision-free LLM answers MCQs
based solely on the generated caption, and its accuracy is the reward.

Paper (§2.1 - Verifiable Reward):
    R(cap) = Accuracy(LLM(cap, MCQs))
    where LLM is a VISION-FREE language model (no image access)

This is the key innovation: reward is VERIFIABLE (MCQ has definite answer)
and REFERENCE-FREE (no human-written reference caption needed).
"""
import json
import re
import random
from typing import Optional


class MCQScorer:
    """
    Vision-free MCQ scorer — simulates the LLM that evaluates caption utility.

    Paper §2.1:
        - Given: caption c, MCQ question q, choices {a1,...,ak}
        - LLM reads c (no image!) and selects the most likely answer
        - R = mean accuracy over all MCQs for the image

    In production: a real LLM (e.g., Qwen2.5-7B) is used to answer MCQs.
    This toy: simple keyword matching to simulate LLM comprehension.
    """

    def __init__(self, model_type: str = "keyword_match"):
        self.model_type = model_type

    def _keyword_answer(self, caption: str, question: str, choices: list[str]) -> str:
        """
        Simulate LLM reading the caption and choosing the best answer.

        A good caption contains the answer explicitly or inferably.
        """
        caption_lower = caption.lower()

        # Score each choice by its mention in the caption
        scores = {}
        for choice in choices:
            # Check for exact mention
            exact_count = caption_lower.count(choice.lower())
            # Check for partial/semantic match (simplified)
            words = choice.lower().split("_")
            word_count = sum(1 for w in words if w in caption_lower)
            scores[choice] = exact_count * 2 + word_count

        # Return the choice with highest score (if any) or random fallback
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return random.choice(choices)  # fallback: random guess

    def score_caption(
        self,
        caption: str,
        mcqs: list[dict],
    ) -> dict:
        """
        Score a caption on a set of MCQs.

        Returns: {accuracy, correct, total, per_question_results}
        """
        correct = 0
        results = []

        for mcq in mcqs:
            predicted = self._keyword_answer(caption, mcq["question"], mcq["choices"])
            is_correct = predicted == mcq["answer"]
            if is_correct:
                correct += 1
            results.append({
                "question": mcq["question"],
                "predicted": predicted,
                "answer": mcq["answer"],
                "correct": is_correct,
            })

        accuracy = correct / (len(mcqs) + 1e-8)
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(mcqs),
            "per_question": results,
        }

    def reward(self, caption: str, mcqs: list[dict]) -> float:
        """
        Main reward function for RL training.

        Paper formula: R(cap) = Acc(LLM(cap, MCQs))
        """
        result = self.score_caption(caption, mcqs)
        return result["accuracy"]


if __name__ == "__main__":
    # Sanity check
    scorer = MCQScorer()

    # Good caption: contains most ground truth info
    good_caption = (
        "A red dog is sitting in a park. "
        "The dog appears to be resting on the grass."
    )
    # Bad caption: generic, missing key details
    bad_caption = "An animal is in a place outdoors."

    mcqs = [
        {
            "question": "What color is the dog?",
            "choices": ["red", "blue", "green", "white"],
            "answer": "red",
        },
        {
            "question": "What is the main object in the scene?",
            "choices": ["dog", "cat", "bicycle", "car"],
            "answer": "dog",
        },
        {
            "question": "What is the dog doing?",
            "choices": ["sitting", "running", "eating", "playing"],
            "answer": "sitting",
        },
        {
            "question": "Where does the scene take place?",
            "choices": ["park", "street", "kitchen", "beach"],
            "answer": "park",
        },
    ]

    print("Good caption score:", scorer.reward(good_caption, mcqs))
    print("Bad caption score: ", scorer.reward(bad_caption, mcqs))
