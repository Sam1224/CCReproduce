# A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems |
| **Authors** | (See arxiv) |
| **Affiliations** | (Industry, SIGIR 2026) |
| **arXiv** | [2605.09338](https://arxiv.org/abs/2605.09338) |
| **Submitted** | 2026-05-10 |
| **Venue** | SIGIR 2026 (July 20–24, Melbourne) |
| **Bucket** | STRONG |
| **Total** | **66 / 100** |

---

## 方法概述 / Method Overview

### EN
Industrial recommendation systems struggle to leverage the rich semantic signals in multimedia content (short videos, product images, live streams) because Multimodal Large Language Models (MM-LLMs) are too slow for real-time inference at scale. This paper proposes a **tripartite framework** for MM-LLM-driven multimedia understanding: (1) **Content Interpretation** — an MM-LLM (LLaMA2-based) generates rich descriptive captions from multimedia content; (2) **Representation Extraction** — captions are tokenized and converted to categorical features for downstream models; (3) **Pipeline Integration** — the derived features are integrated into the existing recommendation training and serving pipeline with offline captioning to avoid latency. This design decouples the heavy MM-LLM inference from real-time serving, enabling industrial-scale deployment.

### ZH
工业级推荐系统难以实时利用多媒体内容（短视频、商品图片、直播）中的丰富语义信号，因为 MM-LLM 推理太慢。本文提出**三部曲框架**：(1) **内容解读**：MM-LLM（基于 LLaMA2）为多媒体内容生成丰富描述性标注；(2) **表示提取**：标注被 tokenize 并转化为下游模型的类别特征；(3) **流水线集成**：离线标注与在线推荐流水线解耦，避免实时延迟。该设计将重型 MM-LLM 推理与实时服务分离，支持工业级部署。

---

## 故事弧 / Story Arc

> **"MM-LLM 语义理解强但实时推理太慢"** → 离线标注+特征离散化，将 MM-LLM 语义注入在线推荐系统，兼顾语义质量与实时性。

---

## 创新性分析 / Innovation Analysis

1. **离线-在线解耦**：离线 MM-LLM 标注 + 在线特征服务，是 MLE 工程上的重要实践路径。
2. **LLM 语义 → 类别 ID**：将 LLM 生成的文本标注 tokenize 为 categorical features，契合现有推荐系统架构。
3. **通用框架**：不依赖特定 MM-LLM，可替换为任意多模态模型。
4. 创新点相对渐进；核心贡献在于工程实践经验而非方法创新。

---

## 关键指标 / Key Metrics

| Metric | Result |
|--------|--------|
| Recall (large-scale deployment) | Improvement over baseline (specific numbers not public) |
| Latency constraint | Met via offline captioning |

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 18 | 工程实践框架，方法创新度中等 |
| Experimental SOTA delta | 15 | 8 | 工业部署验证，数字不够详细 |
| Experimental quality / ablations | 15 | 8 | 大规模部署，但 ablation 不足 |
| Efficiency | 10 | 7 | 离线标注有效解耦 |
| Generalization | 5 | 3 | 框架通用，但实例化为推荐系统 |
| Domain relevance | 25 | 22 | 大规模推荐系统的多媒体理解 |
| **Total** | **100** | **66** | |
