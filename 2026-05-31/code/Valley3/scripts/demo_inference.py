#!/usr/bin/env python
"""Valley3 inference demo: thinking modes + agentic search."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.thinking_modes import ControllableReasoner, ThinkingConfig
from src.agentic_search import AgenticSearchAgent, ToyKnowledgeBase
import torch

def demo_thinking_modes():
    print("=" * 60)
    print("Demo: Controllable Reasoning Modes (Valley3)")
    print("=" * 60)
    reasoner = ControllableReasoner()
    question = "Does this product '健康瘦身茶' claim weight loss of 10kg in 7 days?"

    for level in range(4):
        print(f"\n--- Think Level {level} ---")
        result = reasoner.generate(question, think_level=level)
        if result["thinking"]:
            print(f"  <think> {result['thinking'][:100]}... </think>")
            print(f"  Think tokens used: ~{result['think_tokens']}")
        print(f"  <answer> {result['answer']} </answer>")

def demo_agentic_search():
    print("\n" + "=" * 60)
    print("Demo: Agentic Search (Valley3)")
    print("=" * 60)
    agent = AgenticSearchAgent(ToyKnowledgeBase())

    queries = [
        "Check compliance for product P001 slimming tea efficacy claims",
        "What policy rules apply to influencer disclosure?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        result = agent.execute(q)
        print(f"Agent Trace: {len(result['trace'])} steps")
        for step in result["trace"]:
            print(f"  Step {step['step']}: [{step['tool']}] "
                  f"{step.get('reasoning', '')[:60]}")
        print(f"Answer (excerpt): {result['answer'][:150]}...")

def demo_omni_forward():
    print("\n" + "=" * 60)
    print("Demo: Omni MLLM Forward Pass")
    print("=" * 60)
    from src.omni_model import OmniEcommerceModel
    model = OmniEcommerceModel(vocab_size=256, d_model=128)
    model.eval()

    B, T_text, d = 2, 8, 128
    text_embed = torch.randn(B, T_text, d)
    img_feat = torch.randn(B, 16, 64)
    mel = torch.randn(B, 64, 80)
    input_ids = torch.randint(0, 256, (B, 5))

    with torch.no_grad():
        logits = model(input_ids, text_embed, img_feat, mel)
    print(f"Input: text_embed {text_embed.shape}, img {img_feat.shape}, mel {mel.shape}")
    print(f"Output logits: {logits.shape}  (B={B}, T_dec=5, vocab=256)")
    print("Omni forward pass OK ✓")

if __name__ == "__main__":
    demo_omni_forward()
    demo_thinking_modes()
    demo_agentic_search()
