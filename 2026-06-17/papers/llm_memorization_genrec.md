# On the Memorization Behavior of LLMs in Generative Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | On the Memorization Behavior of LLMs in Generative Recommendation: Observations, Implications, and Training Strategies |
| **Authors** | Sunwoo Kim, Sunkyung Lee, Clark Mingxuan Ju, Donald Loveland, Bhuvesh Kumar, Kijung Shin, Neil Shah, Liam Collins |
| **Affiliations** | KAIST; Sungkyunkwan University; Snap Inc. |
| **arXiv** | — |
| **Submitted** | 2026-06-17 window |
| **Domain Tags** | generative recommendation, LLM memorization, training strategy, long-tail, Snap |

---

## 方法概述 / Method Summary

首次系统研究 LLM 生成式推荐（GenRec）中的"一跳记忆"（One-hop Memorization）现象：模型仅凭记忆训练集中的高频用户-物品交互即可获得高 Recall，而非真正的推理泛化，导致非记忆受益用户（长尾用户）推荐质量差。本文提出即插即用的多任务训练策略 IIRG（In-context Item Recommendation Generation）：增加协同邻居生成任务（Collaborative Neighbor Generation）和语义邻居生成任务（Semantic Neighbor Generation），让模型同时学习"生成邻居"而非仅"记忆交互"，改善长尾用户的推荐质量。

**Story arc**: LLM GenRec 的高性能部分来自训练集记忆而非泛化 → 诊断一跳记忆现象 → IIRG 多任务训练提升真正的泛化性。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 25 | 30 | 首次量化 GenRec 记忆现象，IIRG 多任务创新 |
| Experimental SOTA delta | 12 | 15 | 17 基线对比，SID/TID 双指标 |
| Experimental quality / ablations | 13 | 15 | 扎实消融实验 |
| Efficiency | 6 | 10 | 无效率专项分析 |
| Generalization | 4 | 5 | 跨数据集验证 |
| Domain relevance | 14 | 25 | GenRec 改善，与内容治理弱相关 |
| **Total** | **74** | **100** | 洞察清晰，实验扎实，但与创作者治理直接性弱 |
