"""
Valley3 Main Training Script
Runs the four-stage omni e-commerce pre-training pipeline.

Usage:
  python train.py --stage 1 --epochs 2 --batch_size 2
  python train.py --stage 3 --epochs 3 --batch_size 4 --lr 5e-5
  python train.py --all_stages   # runs all 4 stages sequentially
"""

import argparse
import torch
import os

from model.valley3 import Valley3Config
from training.stage1_audio import stage1_train
from training.stage3_ecom import stage3_train


def get_cfg(args) -> Valley3Config:
    # Toy config for reproduction (full: 7B or 72B LLM backbone)
    return Valley3Config(
        llm_hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        reasoning_mode=args.reasoning_mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Valley3 Training Pipeline")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training stage (1-4)")
    parser.add_argument("--all_stages", action="store_true",
                        help="Run all 4 stages sequentially")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="LLM hidden size (toy: 512, full: 4096)")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--reasoning_mode", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="0=no-think, 1/2/3=increasing depth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = get_cfg(args)
    device = args.device

    print(f"Valley3 Training | Stage: {args.stage} | Device: {device}")
    print(f"Config: hidden={cfg.llm_hidden_size}, heads={cfg.num_heads}, layers={cfg.num_layers}")

    stages_to_run = [1, 2, 3, 4] if args.all_stages else [args.stage]

    for stage in stages_to_run:
        print(f"\n{'='*60}")
        print(f"Starting Stage {stage}")
        print(f"{'='*60}")

        if stage == 1:
            model = stage1_train(cfg, args.epochs, args.batch_size, args.lr, device)
            ckpt_path = os.path.join(args.output_dir, "stage1.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Stage 1 checkpoint saved: {ckpt_path}")

        elif stage == 2:
            # Stage 2: Cross-modal instruction following
            # Full implementation: joint text/image/audio/video instruction tuning
            # Pseudocode:
            #   dataset = CrossModalInstructionDataset(stage=2)
            #   trainer = Stage2Trainer(model, unfreeze_vision=True)
            #   trainer.train(dataset, epochs=args.epochs)
            print("[Stage 2] Cross-modal instruction tuning (pseudocode).")
            print("  → Load Stage 1 checkpoint")
            print("  → Unfreeze vision encoder + projector")
            print("  → Train on interleaved image-text instruction data")
            print("  → Data: ShareGPT4V + LLaVA-Instruct + e-com image-text pairs")

        elif stage == 3:
            trainer = stage3_train(cfg, args.epochs, args.batch_size, args.lr * 0.5, device)
            ckpt_path = os.path.join(args.output_dir, "stage3.pt")
            torch.save(trainer.model.state_dict(), ckpt_path)
            print(f"Stage 3 checkpoint saved: {ckpt_path}")

        elif stage == 4:
            # Stage 4: Long-context reasoning
            # Full implementation: multi-turn e-commerce research dialogues
            # Pseudocode:
            #   dataset = LongContextEcomDataset(stage=4, max_len=32768)
            #   trainer = Stage4Trainer(model, enable_agentic_search=True)
            #   trainer.train(dataset, epochs=args.epochs)
            print("[Stage 4] Long-context reasoning (pseudocode).")
            print("  → Load Stage 3 checkpoint")
            print("  → Extend context window to 32K via RoPE scaling")
            print("  → Train on multi-turn e-commerce deep research dialogues")
            print("  → Integrate agentic search tool-use SFT data")

    print("\nTraining pipeline complete.")

    # Demo agentic search
    if args.reasoning_mode > 0:
        print(f"\nDemonstrating Agentic Search (reasoning_mode={args.reasoning_mode})...")
        from agentic.agentic_search import AgenticSearchLoop, demo_mock_model
        loop = AgenticSearchLoop(demo_mock_model, max_turns=5)
        result = loop.run("Check this product for health claim violations.")
        print(f"Agentic result: {result['final_answer'][:100]}...")


if __name__ == "__main__":
    main()
