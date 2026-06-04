# RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift |
| **Authors** | Kilho Son, Paul Hsu, Cha Zhang, Dinei Florencio |
| **Affiliation** | Microsoft |
| **arXiv** | https://arxiv.org/abs/2606.01697 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | Vector Embeddings · Conversational Retrieval · RAG · Dense Retrieval |
| **Code** | — |

---

## 方法概述 / Method Overview

### 问题 / Problem
多轮对话检索（conversational dense retrieval）中存在**分布偏移（distributional shift）**问题：训练时使用的对话 query 分布与部署时用户真实 query 分布存在差异，导致现有嵌入模型在实际应用中性能大幅下降。现有方法通常需要昂贵且难以获取的**对话式 query-to-document 相关性标注**。

In conversational dense retrieval, **distributional shift** causes embedding models trained on specific query distributions to degrade on real user queries. Existing approaches typically require expensive and hard-to-obtain **conversational query-to-document relevance annotations**.

### 方法 / Method

RCEM 将 LLM 的 query 改写能力**蒸馏（distill）**到嵌入模型中，实现在推理时无需显式 query 改写即可进行上下文感知的检索：

1. **对齐策略：** RCEM 通过将**对话式 query 嵌入**与**改写后 query 嵌入**对齐，而不是直接学习对话式 query 到文档的匹配，从而将 LLM 的语言理解能力注入嵌入空间。

2. **无需 query-doc 相关性标注：** 训练只需 (对话式 query, 改写后 query) 对，避免了对话语境下高成本的相关性标注。

3. **推理时无改写开销：** 蒸馏完成后，嵌入模型本身已具备对话上下文感知能力，推理时无需调用 LLM 做改写。

**Story Arc:** "对话检索需要 LLM 改写但推理成本高 → RCEM 将改写能力蒸馏进嵌入模型，推理零改写开销"

*Conversational retrieval requires LLM rewriting but high inference cost → RCEM distills rewriting skill into embedder, zero overhead at inference.*

---

## 创新性分析 / Innovation

1. **知识蒸馏新范式**：首次将 LLM 的 query 改写能力蒸馏到 embedding 模型，而不是直接监督 query-doc 对，降低了对标注数据的依赖。
2. **分布偏移鲁棒性**：显式针对 distributional shift 设计，在 OOD 设置下实现最高 20% 的提升，具有重要实用价值。
3. **推理效率**：相比需要先调用 LLM 改写再检索的两步方案，RCEM 将两步合一，显著降低延迟。
4. **无需昂贵标注**：不需要对话式 query-document 相关性标注，大幅降低数据工程成本。

---

## 关键指标 / Key Metrics

| Dataset | Metric | RCEM | Baseline |
|---------|--------|------|---------|
| QReCC | MRR@10 | Reported improvement | Best prior |
| TopiOCQA | MRR@10 | Reported improvement | Best prior |
| TREC CAsT | nDCG@3 | Up to +20% under dist. shift | Prior SOTA |

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 22 | Novel distillation of LLM rewriting into embedder; no query-doc annotation needed; distributional shift focus |
| SOTA Delta | 15 | 12 | Up to 20% improvement on QReCC/TopiOCQA under distributional shift |
| Exp. Quality | 15 | 12 | Three benchmarks + explicit distributional shift analysis; ablations |
| Efficiency | 10 | 8 | Zero inference-time LLM rewriting overhead |
| Generalization | 5 | 4 | Three diverse conversational retrieval benchmarks |
| Domain Relevance | 25 | 17 | Vector embeddings + RAG directly relevant; e-commerce search and AI assistant retrieval applications |
| **Total** | **100** | **75** | |

---

## 与先前工作的对比 / Comparison with Prior Work

| Work | Rewriting at Inference | Training Data | Distributional Shift |
|------|----------------------|---------------|---------------------|
| CONQRR, COTED | Yes (LLM rewrite) | query-doc pairs | Not addressed |
| ConvDR | No | query-doc pairs | Partially |
| **RCEM** | **No** | **query-rewrite pairs only** | **Explicitly addressed (+20%)** |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

RCEM 的技术直接适用于：
- **电商 AI 客服的多轮检索**：用户在对话中逐步描述商品需求，RCEM 可在无额外 LLM 调用的情况下准确理解对话上下文，检索相关商品
- **RAG 知识库检索**：内容治理系统中的政策文档检索、违规案例库检索
- **达人搜索**：基于用户多轮交互的达人/内容检索

RCEM applies to: multi-turn e-commerce AI assistant product search, RAG-based policy document retrieval for content governance, and influencer search based on multi-turn user interaction.
