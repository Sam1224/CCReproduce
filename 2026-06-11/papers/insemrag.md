# InSemRAG: Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking |
| **Authors** | (Multi-author) |
| **Affiliation** | (Multi-institution) |
| **ArXiv** | [2606.01240](https://arxiv.org/abs/2606.01240) |
| **Submitted** | June 1, 2026 |
| **Domain Tags** | `RAG` `retrieval` `LLM` `efficiency` `chunking` |
| **Total** | **56 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 传统 RAG 系统存在两大问题：（1）意图无关的检索（intent-agnostic retrieval）无法根据用户真实意图匹配文档；（2）信息碎片化（information fragmentation）——固定长度 chunking 割裂语义完整的段落，导致检索质量下降。

**解决方案：** InSemRAG 框架提出意图感知检索器（intention-aware retriever）和语义保留 chunking（semantics-preserving chunking），解决信息不足问题。

---

## 方法概述 / Method

1. **Intent-Aware Retrieval：** 将用户查询分解为意图感知的子查询，针对不同意图层次进行检索
2. **Semantics-Preserving Chunking：** 基于语义边界而非固定长度切分文档，保留语义完整性
3. **Integration：** 两者结合，提升 RAG 整体效果

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| RAG benchmarks | Answer quality | Improved vs. naive RAG |
| Chunking | Semantic integrity | Better than fixed-length |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 15 | 30 | Intent-aware + semantic chunking combination; incremental improvement |
| Experimental SOTA delta | 9 | 15 | Improved over baseline RAG |
| Experimental quality / ablations | 10 | 15 | Standard RAG evaluation |
| Efficiency | 8 | 10 | Improved retrieval efficiency |
| Generalization | 4 | 5 | General RAG framework |
| Domain relevance | 10 | 25 | RAG applicable to e-commerce knowledge bases and product search |
| **Total** | **56** | **100** | |

---

## 中文摘要

InSemRAG 针对传统 RAG 系统的两大不足：意图无关检索和信息碎片化。提出意图感知检索器（将查询分解为意图子查询）和语义保留 chunking（基于语义边界切分）。对于电商场景中的商品知识库问答、政策文档检索等 RAG 应用有参考价值。
