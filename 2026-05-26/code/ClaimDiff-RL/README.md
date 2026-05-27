# ClaimDiff-RL — Toy Reproduction (PyTorch)

This folder is a **toy but runnable scaffold** faithful to:

**"ClaimDiff-RL: Fine-Grained Caption Reinforcement Learning through Visual Claim Comparison"**  
arXiv: https://arxiv.org/abs/2605.20278  
Authors: Tianle Li, Xuyang Shen, Yan Ma, et al. (CUHK, MiniMax)  
Score: **81 / 100** (STRONG — direct relevance to e-commerce content captioning & influencer governance)

---

## Paper Summary

ClaimDiff-RL addresses the **reward granularity problem** in image caption RL: existing methods judge captions as whole sequences, masking errors at the individual visual-claim level.

**Core idea:** Use reference-conditioned **atomic claim differences** as the reward unit:
1. A multimodal judge (ClaimDiff Judge) enumerates visually-grounded differences between candidate and reference captions.
2. Each difference is verified against the image → classified as hallucination or omission.
3. Open-vocabulary error types and severity levels are assigned.
4. Per-difference statistics are composed into a decomposed reward: `R = -α·R_hall - β·R_omit`.

This allows independent tuning of hallucination vs. omission penalties, enabling researchers to navigate the **Pareto frontier** between faithfulness and coverage.

**Key result:** Surpasses Gemini-3-Pro-Preview on several fine-grained captioning dimensions.

---

## Files

| File | Description |
|------|-------------|
| `data.py` | Toy captioning dataset (COCO-format compatible interface) |
| `judge.py` | ClaimDiff Judge (toy: word-diff + keyword matching; paper: MLLM-based) |
| `reward.py` | Reward composition: R_hall + R_omit + R_length |
| `model.py` | Toy GPT-2-style caption policy; vocab + image feat utilities |
| `train.py` | ClaimDiff-RL training via GRPO with group-normalized advantages |
| `evaluate.py` | Evaluation: hallucination rate, omission rate, F-C balance |
| `requirements.txt` | Dependencies |

---

## Quickstart

```bash
cd CCReproduce/2026-05-26/code/ClaimDiff-RL
pip install -r requirements.txt

# Train (30 epochs, ~10s on CPU)
python train.py --epochs 30 --alpha 1.0 --beta 1.0 --group_size 4

# Train with high hallucination penalty (conservative, complete captions)
python train.py --epochs 30 --alpha 2.0 --beta 0.5 --save checkpoints/hall_focus.pt

# Train with high omission penalty (verbose, covering captions)
python train.py --epochs 30 --alpha 0.5 --beta 2.0 --save checkpoints/omit_focus.pt

# Evaluate
python evaluate.py --ckpt checkpoints/claimdiff_rl.pt
```

---

## Faithfulness to Paper

| Component | Paper | This Toy |
|-----------|-------|----------|
| ClaimDiff Judge | MLLM-based NLI decomposition + visual grounding | Word-diff + keyword matching against `image_desc` |
| Policy model | Multimodal LLM (LLaVA-style) | GPT-2-style LM conditioned on text image feat |
| RL algorithm | GRPO with group-normalized advantages | Same (simplified REINFORCE equivalent) |
| Reward: R_hall | -α · Σ severity(hallucinated claims) | Same formula |
| Reward: R_omit | -β · Σ severity(omitted claims) | Same formula |
| Evaluation | 160-image human-labeled benchmark + VQA | 5-sample toy diagnostic |

**To extend:** Replace `text_to_image_feat()` with a real CLIP/SigLIP encoder, replace the word-diff judge with an LLM-based decomposer + MLLM verifier, and swap the toy LM with a real MLLM policy.

---

## E-commerce / Influencer Governance Applications

- **Product image captioning QC**: use ClaimDiff-RL to train captioning models that minimize hallucinated product attributes (wrong color, material) and missing key features.
- **Short video auto-caption**: apply to influencer content captioning to reduce false claims about products/promotions while ensuring all salient facts are covered.
- **Content moderation description**: train description generators for moderation reports that faithfully describe violations without fabricating evidence.
