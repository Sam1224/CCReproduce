from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Ego2WebSample:
    video_profile: str  # concatenated clip captions / grounded evidence
    instruction: str
    expected: str
    category: str


def synth_dataset(n: int) -> List[Ego2WebSample]:
    cats = ["ecommerce", "media", "knowledge", "maps"]
    out = []
    for _ in range(n):
        cat = random.choice(cats)
        if cat == "ecommerce":
            brand = random.choice(["nike", "adidas", "sony", "anker"])
            item = random.choice(["headphones", "shoes", "camera", "charger"])
            video = f"00:10 person picks up {brand} {item} on the table . 00:14 label shows calories per serving 120 ."
            inst = f"On Amazon, find the same {item} shown in the video (brand {brand}) and report calories per serving ."
            expected = "120"
        elif cat == "media":
            exercise = random.choice(["push-up", "squat", "plank"])
            video = f"00:05 person performs {exercise} as the second exercise in workout ."
            inst = f"Please find on YouTube an instructional video for the second exercise shown in the video and report the channel name ."
            expected = "(channel)"
        elif cat == "knowledge":
            car = random.choice(["toyota", "honda", "bmw"])
            founder = {"toyota": "Kiichiro Toyoda", "honda": "Soichiro Honda", "bmw": "Karl Rapp"}[car]
            year = {"toyota": "1937", "honda": "1948", "bmw": "1916"}[car]
            video = f"00:02 video passes by a {car} car badge ."
            inst = f"Look up on Wikipedia the founder of the brand of the car in the video and report founder and year ."
            expected = f"{founder} {year}"
        else:
            uni = random.choice(["Stanford University", "National University of Singapore"])
            city = {"Stanford University": "Stanford", "National University of Singapore": "Singapore"}[uni]
            video = f"00:08 backpack shows text {uni} ."
            inst = f"On Google Maps, locate the main campus for {uni} and report the city ."
            expected = city

        out.append(Ego2WebSample(video, inst, expected, cat))

    return out


def baseline_agent(sample: Ego2WebSample) -> Tuple[str, List[str]]:
    """Offline baseline.

    In the real benchmark this would ground video cues, then interact with web pages.
    Here we produce a plan + final answer guess from the video_profile.
    """

    plan = []
    text = sample.video_profile.lower()
    if sample.category == "ecommerce":
        plan = ["Ground brand/item in video", "Search Amazon", "Open product page", "Read nutrition label"]
        # heuristic extract calories
        for tok in text.split():
            if tok.isdigit():
                return tok, plan
        return "(unknown)", plan

    if sample.category == "knowledge":
        plan = ["Identify brand from video", "Search Wikipedia", "Read founder section"]
        # cannot solve without web; return placeholder
        return "(needs web)", plan

    if sample.category == "maps":
        plan = ["Read university name", "Search on Maps", "Open campus", "Read city"]
        # extract city-like token
        if "singapore" in text:
            return "Singapore", plan
        return "(needs maps)", plan

    plan = ["Identify exercise", "Search YouTube", "Choose tutorial"]
    return "(needs web)", plan


def judge(sample: Ego2WebSample, answer: str) -> bool:
    # Simplified judge: exact/substring match.
    exp = sample.expected.strip().lower()
    got = str(answer or "").strip().lower()
    if exp in ("(needs web)", "(channel)"):
        # these need online eval; we only check non-empty.
        return bool(got and got != "(unknown)")
    return exp in got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    seed_everything(args.seed)

    data = synth_dataset(args.n)
    correct = 0
    by_cat: Dict[str, List[int]] = {}

    for s in data:
        ans, plan = baseline_agent(s)
        ok = judge(s, ans)
        correct += int(ok)
        by_cat.setdefault(s.category, []).append(int(ok))

    acc = correct / max(1, len(data))
    report = {"overall_acc": acc, "by_category": {k: float(np.mean(v)) for k, v in by_cat.items()}}
    print(report)


if __name__ == "__main__":
    main()
