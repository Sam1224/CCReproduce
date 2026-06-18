# MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation |
| **Authors** | Zehang Wei, Jiaxin Dai, Jiamin Yan, Xiang Xiang |
| **Affiliations** | Huazhong University of Science and Technology (HUST) |
| **arXiv** | — |
| **Submitted** | 2026-06-17 window |
| **Domain Tags** | RAG, outlier detection, energy-based, multi-modal, hallucination |

---

## 方法概述 / Method Summary

RAG 系统在多模态内容质量评估场景中常因检索噪声和分布外（OOD）文档而产生幻觉。MODE-RAG 从流形视角诊断 RAG 检索空间中的离群点（Manifold Outlier Diagnosis），利用变分自由能（Variational Free Energy）构建能量基门控机制（Energy-based Gate），动态控制哪些检索文档进入生成阶段；结合多智能体架构，各 Agent 负责不同模态的内容质量治理。

**Story arc**: RAG 的噪声文档导致生成质量下降 → 流形离群点诊断 + 能量基门控，过滤低质量检索，提升 RAG 在内容质量治理中的可靠性。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | 流形离群点诊断 + 变分自由能门控，理论创新 |
| Experimental SOTA delta | 10 | 15 | 实证覆盖窄，单一 7B kernel |
| Experimental quality / ablations | 11 | 15 | 自建基准，SOTA 优势不稳 |
| Efficiency | 7 | 10 | 多智能体有开销 |
| Generalization | 2 | 5 | 局限于自建基准 |
| Domain relevance | 16 | 25 | 多模态内容质量/幻觉治理有借鉴价值 |
| **Total** | **70** | **100** | 理论新颖但实证偏弱，无开源 |
