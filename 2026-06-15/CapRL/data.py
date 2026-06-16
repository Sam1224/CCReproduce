"""
CapRL++ — Toy Dataset

Generates synthetic image descriptions + MCQs for the RLVR captioning task.

The paper uses real images (ShareGPT4V, BLIP-558K, Open-LLaVA-NeXT-1M).
This toy uses structured text descriptions simulating visual scenes.

Paper: https://arxiv.org/abs/2606.09393
"""
import json
import os
import random

# Simulated visual scenes (stand-in for actual images)
SCENE_TEMPLATES = [
    {
        "scene_id": "scene_{i}",
        "description": "A {color} {object} {action} in a {setting}",
        "elements": {
            "color": ["red", "blue", "green", "white", "black", "yellow"],
            "object": ["dog", "cat", "bicycle", "car", "person", "bird"],
            "action": ["sitting", "running", "standing", "eating", "playing"],
            "setting": ["park", "street", "kitchen", "garden", "beach"],
        },
    }
]


def generate_scene(scene_id: int) -> dict:
    """Generate a toy visual scene with ground truth attributes."""
    tmpl = SCENE_TEMPLATES[0]
    elements = tmpl["elements"]
    attrs = {k: random.choice(v) for k, v in elements.items()}

    description = tmpl["description"].format(**attrs)
    full_desc = (
        f"{description.capitalize()}. "
        f"The {attrs['object']} is {attrs['color']} colored. "
        f"The scene takes place in a {attrs['setting']}."
    )

    # Generate MCQs from the ground truth attributes
    mcqs = generate_mcqs(attrs, scene_id)

    return {
        "scene_id": f"scene_{scene_id:04d}",
        "ground_truth_attrs": attrs,
        "full_description": full_desc,
        "mcqs": mcqs,
    }


def generate_mcqs(attrs: dict, scene_id: int) -> list[dict]:
    """
    Generate Multiple Choice Questions from scene attributes.

    Paper (§2): MCQs are generated from the image to evaluate caption utility.
    A good caption should allow a vision-free LLM to answer these MCQs.
    """
    mcqs = []
    elements = SCENE_TEMPLATES[0]["elements"]

    q_templates = [
        ("What color is the {object}?", "color"),
        ("What is the main object in the scene?", "object"),
        ("What is the {object} doing?", "action"),
        ("Where does the scene take place?", "setting"),
    ]

    for q_tmpl, target_key in q_templates:
        correct = attrs[target_key]
        wrong_choices = [v for v in elements[target_key] if v != correct]
        distractors = random.sample(wrong_choices, k=min(3, len(wrong_choices)))
        all_choices = [correct] + distractors
        random.shuffle(all_choices)

        mcqs.append({
            "question": q_tmpl.format(**attrs),
            "choices": all_choices,
            "answer": correct,
            "target_attr": target_key,
        })

    return mcqs


def build_dataset(output_dir: str = "data", n_scenes: int = 500):
    os.makedirs(output_dir, exist_ok=True)

    scenes = [generate_scene(i) for i in range(n_scenes)]

    # Split train/val/test
    random.shuffle(scenes)
    n_train = int(0.8 * n_scenes)
    n_val = int(0.1 * n_scenes)
    splits = {
        "train": scenes[:n_train],
        "val": scenes[n_train:n_train + n_val],
        "test": scenes[n_train + n_val:],
    }

    for split, data in splits.items():
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Dataset: {n_scenes} scenes → train={n_train}, val={n_val}, test={len(scenes)-n_train-n_val}")
    return splits


if __name__ == "__main__":
    build_dataset()
