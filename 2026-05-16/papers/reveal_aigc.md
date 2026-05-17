# Reasoning-Aware AIGC Detection via Alignment and Reinforcement (REVEAL)

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Reasoning-Aware AIGC Detection via Alignment and Reinforcement |
| **Authors** | (Research team) |
| **Affiliation** | (University / Research lab) |
| **arXiv ID** | [2604.19172](https://arxiv.org/abs/2604.19172) |
| **Submission Date** | April 2026 |
| **Domain Tags** | `#AIGC-detection` `#content-governance` `#LLM` `#reasoning` `#RLHF` `#benchmark` |
| **Total** | **66 / 100** |

---

## 故事弧线 / Story Arc

**现有问题:** 随着 LLM 能力持续增强，AI 生成文本（AIGC）与人类写作日益难以区分。现有 AIGC 文本检测器普遍是**黑盒**分类器，缺乏可解释的推理链，无法给出"为什么认为这是 AI 生成"的透明依据，在域外迁移和对抗挑战下泛化性不足。

**设计方案:** 提出 **REVEAL（Reasoning and Evaluation of AIGC through Aligned Language）**：在分类前显式建模 **Think-then-Answer** 推理过程，通过两阶段训练（SFT + 强化学习 RL）提升检测精度、逻辑一致性和抗幻觉能力。同时构建大规模多领域 AIGC 文本基准 **AIGC-text-bank**。

---

## 方法概述 / Method Overview

### 两阶段训练

```
Stage 1: Supervised Fine-Tuning (SFT)
   ├── Train model to generate reasoning chain BEFORE label
   ├── Reasoning: "This text shows [evidence], which is typical of AI because [reason]..."
   └── Establishes Think-then-Answer capability

Stage 2: Reinforcement Learning (RL)
   ├── Reward: accuracy (correct label) + logical consistency (chain validity)
   ├── Penalty: hallucinated evidence, circular reasoning
   └── Reduces hallucinations, improves reasoning quality
```

### AIGC-text-bank
- 大规模多领域数据集，覆盖最新 SOTA LLM 输出
- 包含多种著作场景：完全 AI 生成、人机协作、人工撰写
- 将开源供社区研究使用

---

## 关键指标 / Key Metrics

| Benchmark | Setting | REVEAL | Black-box Detectors | General LLMs |
|-----------|---------|--------|---------------------|--------------|
| 5 benchmarks | Binary detection | **SOTA** | Lower | Lower |
| 5 benchmarks | Fine-grained detection | **SOTA** | N/A | Lower |
| Domain shift | Generalization | Strong | Degrades | Degrades |
| Adversarial | Robustness | Strong | Weak | Moderate |

---

## 创新性分析 / Innovation Analysis

- Think-then-Answer 范式将 AIGC 检测从黑盒升级为可解释推理
- SFT + RL 两阶段训练提供了减少幻觉的系统化路径
- 在对抗挑战和域外场景下泛化性优于现有黑盒检测器

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 19 | 30 | Think-then-Answer 推理新颖；SFT+RL 组合系统化 |
| Experimental SOTA Delta | 10 | 15 | 5 基准 SOTA；具体数值待访问论文确认 |
| Experimental Quality | 9 | 15 | 多领域基准；对抗+域外评测设计合理 |
| Efficiency | 6 | 10 | 推理链增加延迟，但规模适中 |
| Generalization | 4 | 5 | 域移和对抗场景下显示强泛化 |
| Domain Relevance | 18 | 25 | AIGC 检测是内容治理重要场景，但非直接电商 |
| **Total** | **66** | **100** | |
