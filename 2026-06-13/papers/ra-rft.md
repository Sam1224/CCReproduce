# RA-RFT: Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning |
| **ArXiv ID** | [2606.13680](https://arxiv.org/abs/2606.13680) |
| **Authors** | Zilin Xiao, Qi Ma, Chun-cheng Jason Chen, Xintao Chen, Avinash Atreya, Hanjie Chen, Vicente Ordonez |
| **Affiliation** | Various (multiple US universities) |
| **Submitted** | 2026-06-11 |
| **Source** | HuggingFace June 13 daily listing |
| **Bucket** | WEAK — RAG+RLVR 推理后训练 |

---

## 方法概述 / Method Overview

**故事弧线：** RLVR（基于可验证奖励的推理后训练）在稀疏奖励场景下探索效率低；传统 RAG 按语义相似检索的样例"看起来像"但策略不对，可能干扰训练。→ RA-RFT 定义"推理效用"（reasoning utility）为检索目标，通过 LLM judge 蒸馏训练推理感知 retriever，在 RLVR 训练期注入高质量类比推理轨迹，加速有效探索。

**三步流程：**
1. **Reasoning-Utility 蒸馏**：用 LLM judge 对 query-corpus 对生成推理相关性标签（gold-relevance labels）；
2. **Reasoning-Aware Retriever 训练**：以 utility 标签训练 retriever，使检索到的样例在推理策略层面而非表面语义层面相似；
3. **RLVR 训练注入**：将检索到的推理轨迹作为 few-shot demonstrations 注入 RLVR prompt，提升 rollout 质量与奖励密度。

**关键创新：**
- 将检索目标从"语义相似"转为"推理效用"，提供 reasoning-level 的类比样例；
- 训练期（而非仅推理期）集成检索增强，使模型学会利用类比策略；
- 在竞赛级数学基准上稳定提升。

---

## 关键指标 / Key Metrics

| 模型 | 方法 | AIME24 | AIME25 | HMMT25 | BrUMO25 |
|------|------|--------|--------|--------|---------|
| Qwen3-1.7B | GRPO | 50.4 | 41.6 | 26.3 | 54.8 |
| Qwen3-1.7B | **RA-RFT** | **55.1** | **48.7** | **28.2** | **57.4** |
| Qwen3-4B | GRPO | 74.8 | 66.4 | 46.4 | 69.8 |
| Qwen3-4B | **RA-RFT** | **75.8** | **69.2** | **47.3** | **75.7** |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 23 | 30 | 推理效用检索 + 训练期 RAG 整合，概念清晰且有实证 |
| 实验指标 | 12 | 15 | 多竞赛级数学基准，提升稳定 |
| 实验质量 | 12 | 15 | retriever/优化算法 ablation，多模型规模验证 |
| 方法效率 | 4 | 10 | 训练成本高（retriever 训练 + RLVR） |
| 方法泛化性 | 4 | 5 | 框架通用，但主要验证数学推理 |
| 领域相关性 | 12 | 25 | 通用 RAG+RLVR 范式可借鉴，但非电商/内容直接应用 |
| **Total** | **67** | **100** | |

---

## Story Arc

> **现状不足：** RLVR 在策略稀疏时陷入低效探索；传统 RAG 语义相似检索干扰推理训练。  
> **解法：** 以推理效用为检索目标 → 蒸馏 reasoning-aware retriever → 训练期注入类比轨迹 → 竞赛级数学推理稳定提升。
