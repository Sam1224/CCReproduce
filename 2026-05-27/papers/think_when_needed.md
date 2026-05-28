# Think When Needed: Adaptive Reasoning-Driven Multimodal Embeddings with a Dual-LoRA Architecture

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Think When Needed: Adaptive Reasoning-Driven Multimodal Embeddings with a Dual-LoRA Architecture |
| **Authors** | (See arxiv) |
| **Affiliations** | (See arxiv) |
| **arXiv** | [2605.14448](https://arxiv.org/abs/2605.14448) |
| **Submitted** | 2026-05-14 |
| **Bucket** | WEAK |
| **Total** | **64 / 100** |

---

## 方法概述 / Method Overview

### EN
Recent multimodal embedding models inject chain-of-thought (CoT) reasoning into the embedding pipeline to improve retrieval quality. However, existing approaches always generate CoT regardless of input complexity, using separate reasoner and embedder models with high parameter overhead. For simple inputs, reasoning adds noise rather than signal. **Think When Needed (TWN)** proposes a unified framework with **adaptive reasoning**: a **dual-LoRA architecture** attaches two LoRA adapters — a reasoning adapter and an embedding adapter — to a shared frozen backbone. Gradients at their interface are detached to mitigate conflicts from joint optimization. A lightweight **complexity router** determines whether CoT is needed before generating the embedding, skipping reasoning for simple inputs. This achieves comparable retrieval quality to always-on reasoning systems while reducing inference cost significantly.

### ZH
现有多模态嵌入模型将思维链（CoT）推理注入嵌入流水线，但对所有输入无差别生成 CoT，使用独立推理器+嵌入器，参数开销大。对于简单输入，推理反而引入噪声。**Think When Needed（TWN）** 提出**自适应推理**框架：**双 LoRA 架构**在共享冻结主干上挂载推理 LoRA 和嵌入 LoRA，接口处梯度截断以缓解联合优化冲突。轻量**复杂度路由器**在生成嵌入前判断是否需要 CoT，简单输入跳过推理。在保持与全推理系统相近检索质量的同时，大幅降低推理成本。

---

## 故事弧 / Story Arc

> **"CoT 嵌入推理成本高、简单输入反而有害"** → 双 LoRA 统一架构 + 复杂度路由器，按需推理，参数接近单模型，推理质量接近全推理，成本大幅降低。

---

## 关键指标 / Key Metrics

| Benchmark | TWN vs. always-on CoT | TWN vs. no-CoT |
|-----------|----------------------|----------------|
| Multimodal Retrieval | ~comparable | +significant |
| Inference cost | ~60% reduction | similar |

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 20 | 双 LoRA + 自适应路由思路新颖 |
| Experimental SOTA delta | 15 | 9 | 相对 always-on CoT 持平，相对 no-CoT 提升 |
| Experimental quality / ablations | 15 | 10 | ablation 覆盖路由策略 |
| Efficiency | 10 | 9 | ~60% 推理成本节省是核心卖点 |
| Generalization | 5 | 4 | 多种检索任务 |
| Domain relevance | 25 | 12 | 多模态嵌入通用，电商场景可迁移 |
| **Total** | **100** | **64** | |
