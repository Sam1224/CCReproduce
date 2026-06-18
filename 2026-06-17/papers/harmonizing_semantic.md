# Harmonizing Semantic and Collaborative in LLMs: Reasoning-based Embedding Generator for Sequential Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Harmonizing Semantic and Collaborative in LLMs: Reasoning-based Embedding Generator for Sequential Recommendation |
| **Authors** | Qidong Liu, Mingyao Huang, Moranxin Wang, Wenxuan Yang, Haiping Zhu |
| **Affiliations** | Xi'an Jiaotong University |
| **arXiv** | — |
| **Submitted** | 2026-06-17 window |
| **Domain Tags** | sequential recommendation, LLM embedding, collaborative filtering, RL, long-tail |

---

## 方法概述 / Method Summary

传统推荐 LLM 要么只利用语义信息（忽视协同信号），要么只有 ID-based 协同过滤（忽视语义）。本文提出将 LLM 改造为"推荐 Embedding 生成器"：通过**潜在推理**（Latent Reasoning）让 LLM 在推断物品向量前先进行链式推理；结合**共现奖励 RL**（Co-occurrence Reward RL）将协同过滤中的共现信号作为奖励，驱动 LLM 推理对齐协同模式。生成的 Embedding 可即插即用（离线预计算）与任意推荐 Backbone 结合，无在线额外开销。

**Story arc**: LLM 语义与协同过滤信号长期割裂，融合方案要么端到端成本高、要么对齐不足 → 用 RL 奖励让 LLM 的潜在推理向协同信号靠拢，离线生成高质量混合 Embedding。

**Key metrics**:
- 长尾物品推荐显著提升
- 跨 backbone（SASRec, GRU4Rec）和跨 LLM 泛化稳定

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 25 | 30 | 潜在推理 + 共现奖励 RL 新颖组合 |
| Experimental SOTA delta | 13 | 15 | 多数据集 SOTA，长尾改善突出 |
| Experimental quality / ablations | 13 | 15 | 多 backbone 多 LLM 消融 |
| Efficiency | 9 | 10 | 离线预计算，零在线开销 |
| Generalization | 5 | 5 | 跨 backbone/LLM 泛化 |
| Domain relevance | 12 | 25 | 序列推荐 Embedding，与创作者治理需二次迁移 |
| **Total** | **77** | **100** | 方法新颖，泛化强，但与电商内容治理直接相关性弱 |
