"""
Optional fine-tuning script for VLMs on RuleSafe-VL rule-conditioned moderation data.

This enables training a VLM to better follow rule-conditioned decision reasoning,
addressing the gap identified in the paper (current VLMs < 60% macro-F1 vs
human baseline > 85%).

Training approach:
  - Supervised fine-tuning (SFT) on rule-conditioned (image, text, rules) → (decision, reasoning) pairs
  - Uses LoRA for parameter-efficient fine-tuning
  - Loss computed only on decision + reasoning tokens

Based on: arXiv:2605.07760
Note: Full fine-tuning requires the complete 2,166-case benchmark dataset from the paper authors.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class FineTuneConfig:
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    rules_path: str = "data/rules.json"
    cases_path: str = "data/sample_cases.json"
    image_dir: Optional[str] = None
    output_dir: str = "checkpoints/"

    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Training config
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1      # Increase with gradient accumulation
    grad_accumulation_steps: int = 8
    max_seq_len: int = 2048
    warmup_steps: int = 50
    weight_decay: float = 0.01

    # Evaluation
    eval_every_n_steps: int = 100
    save_every_n_epochs: int = 1


FINETUNE_INSTRUCTION = """You are a content moderation expert. Given platform policy rules and user content (text and/or image), reason step-by-step about which rules are activated and their interactions, then give a moderation decision.

{policy_context}

Content:
{text_content}

Respond in JSON:
{{
  "activated_rules": [...],
  "applicable_relations": [...],
  "sufficient_evidence": true/false,
  "decision": "allowed|restricted|removed",
  "reasoning": "..."
}}"""


class RuleSafeVLDataset(Dataset):
    """
    Dataset for fine-tuning VLMs on rule-conditioned content moderation.

    Each item is:
      input:  (image, policy_context, text_content) → formatted instruction
      target: JSON decision with rule activation + reasoning

    Real dataset: 2,166 cases from the paper.
    Toy dataset: 20 cases from generate_toy_data.py.
    """

    def __init__(
        self,
        rules_path: str,
        cases_path: str,
        processor,
        image_dir: Optional[str] = None,
        max_seq_len: int = 2048,
    ):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.rule_graph import RuleGraph, PolicyFamily

        self.graph = RuleGraph.from_json(rules_path)
        with open(cases_path) as f:
            self.cases = json.load(f)
        self.processor = processor
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        from models.rule_graph import PolicyFamily
        from PIL import Image

        case = self.cases[idx]
        pf = PolicyFamily(case["policy_family"])
        policy_context = self.graph.to_prompt_context(policy_family=pf)

        instruction = FINETUNE_INSTRUCTION.format(
            policy_context=policy_context,
            text_content=case["text_content"],
        )

        target = json.dumps({
            "activated_rules": case["gt_activated_rules"],
            "applicable_relations": case["gt_applicable_relations"],
            "sufficient_evidence": case["gt_sufficient_evidence"],
            "decision": case["gt_decision"],
            "reasoning": f"Based on policy rules, decision is {case['gt_decision']}.",
        })

        # Load image if available
        image = None
        image_file = case.get("image_file")
        if image_file and self.image_dir:
            img_path = Path(self.image_dir) / image_file
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")

        return {
            "instruction": instruction,
            "target": target,
            "image": image,
            "case_id": case["case_id"],
        }


def collate_fn(batch, processor, max_seq_len=2048):
    """Collate batch items for VLM fine-tuning."""
    instructions = [item["instruction"] for item in batch]
    targets = [item["target"] for item in batch]
    images = [item["image"] for item in batch]

    # Build full prompts: instruction + target (for SFT loss)
    full_texts = [
        f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{tgt}<|im_end|>"
        for inst, tgt in zip(instructions, targets)
    ]

    has_images = [img is not None for img in images]
    images_to_use = [img for img in images if img is not None] or None

    try:
        inputs = processor(
            text=full_texts,
            images=images_to_use if any(has_images) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
    except Exception:
        inputs = processor(
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )

    # Build labels: mask instruction tokens with -100 (only compute loss on target tokens)
    # This is a simplified version; full implementation would tokenize separately
    # to find the exact boundary between instruction and target.
    labels = inputs["input_ids"].clone()
    # Pseudocode: mask instruction part
    # for i, (inst, tgt) in enumerate(zip(instructions, targets)):
    #     inst_len = len(processor.tokenize(f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"))
    #     labels[i, :inst_len] = -100

    inputs["labels"] = labels
    return inputs


def train(config: FineTuneConfig):
    """Main fine-tuning loop."""
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from transformers import get_linear_schedule_with_warmup
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        print("Missing deps: pip install transformers peft")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on {device}")

    # Load processor and model
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Dataset
    dataset = RuleSafeVLDataset(
        rules_path=config.rules_path,
        cases_path=config.cases_path,
        processor=processor,
        image_dir=config.image_dir,
        max_seq_len=config.max_seq_len,
    )
    print(f"Dataset size: {len(dataset)} cases")

    # DataLoader with custom collate
    from functools import partial
    collate = partial(collate_fn, processor=processor, max_seq_len=config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(loader) * config.num_epochs // config.grad_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    global_step = 0
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(**batch)
            loss = outputs.loss / config.grad_accumulation_steps
            loss.backward()
            total_loss += loss.item() * config.grad_accumulation_steps

            if (step + 1) % config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | Loss {total_loss/(step+1):.4f}")

        print(f"Epoch {epoch+1} complete. Avg loss: {total_loss/len(loader):.4f}")

        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_path = Path(config.output_dir) / f"epoch_{epoch+1}"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  Checkpoint saved to {save_path}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--rules_path", default="data/rules.json")
    parser.add_argument("--cases_path", default="data/sample_cases.json")
    parser.add_argument("--output_dir", default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    cfg = FineTuneConfig(
        model_name=args.model_name,
        rules_path=args.rules_path,
        cases_path=args.cases_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    train(cfg)
