"""
Valley3 End-to-End Training Script.

Usage:
    python train.py --model_size tiny --stages 1,2,3,4 --toy_data
    python train.py --model_size tiny --stages 3 --toy_data  # Stage 3 only
    python train.py --model_size tiny --stages 1,2,3,4 --epochs_per_stage 2

Paper: Valley3 uses a four-stage omni e-commerce continued pre-training pipeline:
  Stage 1: Audio understanding (audio encoder + A-projector only)
  Stage 2: Cross-modal instruction following (LLM + projectors)
  Stage 3: E-commerce domain knowledge (full fine-tuning)
  Stage 4: Long-context reasoning + controllable thinking (post-training)
"""

import argparse
import torch
import os

from model.valley3 import build_valley3_tiny, Valley3Config, Valley3Model
from training.dataset import get_dataloader
from training.stage1_audio import train_stage1
from training.stage2_crossmodal import train_stage2
from training.stage3_ecom import train_stage3
from training.stage4_longcontext import train_stage4
from evaluation.ecom_benchmark import EcomBenchmark, run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Valley3 Training Pipeline")
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small"],
                        help="Model size (tiny=4 layers for testing)")
    parser.add_argument("--stages", type=str, default="1,2,3,4",
                        help="Comma-separated list of stages to run")
    parser.add_argument("--toy_data", action="store_true", default=True,
                        help="Use toy data (for testing)")
    parser.add_argument("--data_size", type=int, default=50,
                        help="Number of samples per stage (toy mode)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs_per_stage", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--eval_after_training", action="store_true", default=True)
    return parser.parse_args()


def build_model(model_size: str) -> Valley3Model:
    if model_size == "tiny":
        return build_valley3_tiny()
    elif model_size == "small":
        config = Valley3Config(
            vision_hidden_size=512,
            audio_hidden_size=512,
            llm_hidden_size=1024,
            llm_num_layers=8,
            llm_num_heads=16,
            llm_vocab_size=32000,
            num_experts=4,
            num_experts_per_token=2,
        )
        return Valley3Model(config)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def save_checkpoint(model: Valley3Model, stage: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"stage{stage}")
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    print(f"  Checkpoint saved: {path}/model.pt")


def main():
    args = parse_args()

    stages = [int(s.strip()) for s in args.stages.split(",")]
    print("=" * 60)
    print("Valley3 Training Pipeline")
    print(f"  Model: {args.model_size}")
    print(f"  Stages: {stages}")
    print(f"  Data size: {args.data_size} samples/stage")
    print(f"  Epochs/stage: {args.epochs_per_stage}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # Build model
    model = build_model(args.model_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Stage training functions
    stage_trainers = {
        1: train_stage1,
        2: train_stage2,
        3: train_stage3,
        4: train_stage4,
    }

    # Stage-specific learning rates (paper defaults)
    stage_lrs = {1: 2e-4, 2: 1e-4, 3: 5e-5, 4: 2e-5}

    # Run requested stages sequentially
    for stage in stages:
        print(f"\n{'='*40}")
        print(f"STAGE {stage}: {['', 'Audio Understanding', 'Cross-Modal ICF', 'E-commerce Domain', 'Long-Context'][stage]}")
        print(f"{'='*40}")

        dataloader = get_dataloader(
            stage=stage,
            batch_size=args.batch_size,
            size=args.data_size,
        )

        train_fn = stage_trainers[stage]
        model = train_fn(
            model=model,
            dataloader=dataloader,
            num_epochs=args.epochs_per_stage,
            lr=stage_lrs[stage],
            device=args.device,
        )

        save_checkpoint(model, stage, args.save_dir)
        print(f"Stage {stage} complete.")

    # Evaluation
    if args.eval_after_training:
        print(f"\n{'='*60}")
        print("POST-TRAINING EVALUATION on Omni E-commerce Benchmark")
        print(f"{'='*60}")
        benchmark = EcomBenchmark(num_samples_per_task=5)
        results = run_evaluation(model, benchmark, device=args.device)
        print(f"\nFinal Overall Accuracy: {results['overall_accuracy']:.2%}")

    print("\n[Done] Valley3 training pipeline complete.")


if __name__ == "__main__":
    main()
