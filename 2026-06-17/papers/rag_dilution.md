# When More Documents Hurt RAG: Mitigating Vector Search Dilution

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | When More Documents Hurt RAG: Mitigating Vector Search Dilution with Domain-Scoped, Model-Agnostic Retrieval |
| **Authors** | (not fully extracted) |
| **Affiliations** | — |
| **arXiv** | [2606.11350](https://arxiv.org/abs/2606.11350) |
| **Submitted** | ~2026-06-09 |
| **Domain Tags** | RAG, vector search, domain-scoped retrieval, dilution, embedding |

---

## 方法概述 / Method Summary

大规模异构文档集合中，dense similarity 检索随数据规模增大而失去区分力——Top-k 检索越来越多地返回语义相似但上下文错误的块，即"向量搜索稀释"（Vector Search Dilution）问题。本文提出领域作用域检索（Domain-Scoped Retrieval）：基于文档类别/领域先过滤候选集，再做 dense retrieval，显著提升大规模 RAG 的精度，且模型无关（Model-Agnostic）。

**Story arc**: RAG 扩展到大规模文档集后检索精度下降 → 领域作用域过滤 + 密集检索，解决稀释问题。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | 领域作用域过滤思路清晰，但较简单 |
| Experimental SOTA delta | 10 | 15 | 多个大规模基准验证 |
| Experimental quality / ablations | 10 | 15 | 基准测试充分 |
| Efficiency | 7 | 10 | 轻量过滤步骤 |
| Generalization | 4 | 5 | 模型无关，泛化好 |
| Domain relevance | 12 | 25 | RAG 质量改善，可应用于电商知识库 |
| **Total** | **63** | **100** | 实用改进，但创新度偏低 |
