"""
CapRL++ — RLVR Training Loop

Trains the LVLM captioner to maximize MCQ accuracy using REINFORCE.

Paper (§2.3 - RLVR Training):
    L(θ) = -E_{cap ~ π_θ(·|image)}[R(cap, MCQs)]
    ∇_θ L(θ) ≈ -R(cap) · ∇_θ log π_θ(cap|image)  [REINFORCE]

In production (paper §2.3):
    - Algorithm: GRPO (Group Relative Policy Optimization) or PPO
    - Backbone: InternLM2.5-7B or Qwen2.5-VL
    - Dataset: CapRL-Image-1M (1M image-MCQ pairs)
    - Batch size: 64 images × 4 captions per image
"""
import json
import os
import random
import sys


def train(
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    epochs: int = 10,
    lr: float = 0.05,
    output_dir: str = "checkpoints",
):
    os.makedirs(output_dir, exist_ok=True)

    from captioner import ToyLVLMCaptioner
    from mcq_scorer import MCQScorer

    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    captioner = ToyLVLMCaptioner(skill_level=0.3)  # start low
    scorer = MCQScorer()

    print(f"Training CapRL++ ({epochs} epochs, {len(train_data)} train scenes)...")
    print(f"{'Epoch':>5} {'Train R':>8} {'Val R':>8} {'Skill':>7}")
    print("-" * 35)

    best_val_reward = 0.0

    for epoch in range(epochs):
        train_rewards = []
        random.shuffle(train_data)

        for scene in train_data:
            # Step 1: Generate caption (sample from policy)
            caption = captioner.generate_caption(scene)

            # Step 2: Compute verifiable reward (MCQ accuracy)
            reward = scorer.reward(caption, scene.get("mcqs", []))
            train_rewards.append(reward)

            # Step 3: RL update (REINFORCE)
            # In production: compute full gradient with LLM log probs
            # Toy: update skill_level as proxy for gradient step
            baseline = sum(train_rewards[-10:]) / (min(len(train_rewards), 10) + 1e-8)
            advantage = reward - baseline
            captioner.update_skill(reward, lr=lr * max(0.1, abs(advantage)))

        # Validation
        val_rewards = []
        for scene in val_data:
            caption = captioner.generate_caption(scene)
            r = scorer.reward(caption, scene.get("mcqs", []))
            val_rewards.append(r)

        mean_train = sum(train_rewards) / len(train_rewards)
        mean_val = sum(val_rewards) / len(val_rewards)
        print(f"{epoch+1:>5} {mean_train:>8.4f} {mean_val:>8.4f} {captioner.skill_level:>7.4f}")

        if mean_val > best_val_reward:
            best_val_reward = mean_val
            # Save best checkpoint
            ckpt = {"skill_level": captioner.skill_level, "epoch": epoch + 1, "val_reward": mean_val}
            with open(os.path.join(output_dir, "best.json"), "w") as f:
                json.dump(ckpt, f)

    print(f"\nBest validation reward: {best_val_reward:.4f}")
    print(f"Checkpoint: {os.path.join(output_dir, 'best.json')}")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    train(epochs=epochs)
