# Thinking Broad, Acting Fast: Latent Reasoning Distillation for E-Commerce Relevance

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Thinking Broad, Acting Fast: Latent Reasoning Distillation from Multi-Perspective Chain-of-Thought for E-Commerce Relevance |
| **Authors** | Baopu Qiu, Hao Chen, Yuanrong Wu, Changtong Zan, Chao Wei, Weiru Zhang, Xiaoyi Zeng |
| **Venue** | WWW 2026 (ACM Web Conference 2026, Dubai, April 13–17, 2026) |
| **arXiv** | https://arxiv.org/abs/2601.21611 |
| **Submitted** | January 29, 2026 |
| **Domain** | E-commerce Search Relevance · Knowledge Distillation · Chain-of-Thought · LLM |

---

## 方法概述 / Method Summary

电商搜索相关性判断是用户体验的核心。现有 LLM 方法通过 CoT 推理提升相关性判断精度，但存在两个问题：(1) 单视角 CoT 无法覆盖相关性的多维本质（用户意图 vs. 属性匹配 vs. 业务规则）；(2) LLM 推理延迟高，无法满足在线服务的毫秒级要求。

**Thinking Broad, Acting Fast** 提出：
1. **多视角 CoT（Multi-Perspective CoT）：** 从用户意图、商品属性、业务规则等多个维度并行生成 CoT 推理链，覆盖电商相关性的多维本质
2. **隐式推理蒸馏（Latent Reasoning Distillation）：** 将多视角 CoT 蒸馏到紧凑的学生模型中，学生模型在隐式空间（latent space）中内化推理过程，推理时不需要显式生成 CoT 文本
3. 学生模型具备低延迟，可直接部署到在线电商搜索服务

---

## 故事弧线 / Story Arc

> 单视角 CoT 和 LLM 推理延迟制约了电商相关性判断的质量与效率 → 多视角 CoT 覆盖相关性多维本质，隐式推理蒸馏压缩推理到轻量学生模型 → 既提升了相关性判断精度，又满足了在线服务延迟要求

---

## 创新分析 / Innovation Analysis

- 明确提出"多维相关性"的概念并通过多视角 CoT 建模，比单一 CoT 更贴近业务现实
- 隐式推理蒸馏是 KD 领域的创新：不蒸馏 logits，而是将 CoT 推理路径"内化"到 latent 表示中
- 发表于 WWW 2026，同行评审认可

---

## 关键指标 / Key Metrics

| Task | Dataset/Setting | Metric | Result |
|------|----------------|--------|--------|
| E-commerce relevance | E-commerce search benchmark | AUC / F1 | Outperforms single-perspective CoT baselines |
| Latency | Online serving | Inference time | 接近轻量模型水平 |
| Online A/B | Real e-commerce platform | GMV / CTR improvement | 正向提升 |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 多视角 CoT + latent 蒸馏创新 |
| Experimental SOTA delta | 10 | 15 | WWW 2026 录用，超过基线 |
| Experimental quality | 11 | 15 | offline + online A/B |
| Efficiency | 8 | 10 | 蒸馏解决延迟问题 |
| Generalization | 4 | 5 | 通用电商搜索相关性 |
| Domain relevance | 20 | 25 | 电商搜索核心场景 |
| **Total** | **75** | **100** | |

