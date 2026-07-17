import argparse
import json
import random
from pathlib import Path

OBJECTS = ["bus", "table", "cake", "person", "sign", "dog", "cat", "car", "chair"]


def make_record(idx, visual_feature, injected_error=None):
    tags = random.sample(OBJECTS, k=random.randint(1, 3))
    if visual_feature and idx % 3 == 0 and visual_feature not in tags:
        tags[0] = visual_feature
    caption = "The image contains " + ", ".join(tags) + "."
    if injected_error and visual_feature in tags and random.random() < 0.85:
        caption += f" A {injected_error} is visible in the scene."
    features = [1.0 if obj in tags else 0.0 for obj in OBJECTS]
    return {"id": idx, "caption": caption, "image_tags": tags, "image_features": features}


def build_dataset(size=120, visual_feature="table", textual_error="white tablecloth", seed=7):
    random.seed(seed)
    return [make_record(idx, visual_feature, textual_error) for idx in range(size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="toy_symbal.json")
    parser.add_argument("--size", type=int, default=120)
    args = parser.parse_args()
    records = build_dataset(args.size)
    Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
