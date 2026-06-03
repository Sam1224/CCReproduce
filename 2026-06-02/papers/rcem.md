# RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift |
| **Authors** | Kilho Son, Paul Hsu, Cha Zhang, Dinei Florencio |
| **Affiliations** | Microsoft (15010 NE 36th Street, Redmond, WA 98052, US) |
| **Link** | https://arxiv.org/abs/2606.01697 |
| **Submission Date** | 2026-06 (appears in 2026-06-02 listing) |
| **Domain Bucket** | **WEAK** — conversational dense retrieval; transferable to product search with chat-style queries |
| **Total** | **66 / 100** |

---

## 方法概述 / Method Overview

### 问题背景 (Problem)
对话式密集检索（Conversational Dense Retrieval）面临**分布偏移（Distributional Shift）**问题：训练时使用某种对话风格的查询，部署时遇到不同分布（不同用户表达习惯、不同平台上下文），性能急剧下降。

现有方法要么：
1. 依赖**显式查询改写（Explicit Query Rewriting）**，推理时需单独调用 LLM 改写，延迟高
2. 依赖**高质量对话-文档相关性标注**，获取成本高昂

> X is insufficient: Existing conversational retrieval models either need expensive explicit rewriting at inference, or require costly conversation-to-document relevance data.

### 设计 (Design — RCEM)
RCEM（Rewriting-Calibrated Embedding Model）的核心思路：
1. 训练时，使用 LLM 对每条对话查询生成**改写版本（Rewritten Query）**
2. 用对比学习对齐：**对话查询嵌入 ≈ 改写查询嵌入**（知识蒸馏至嵌入层）
3. 推理时，**不需要显式改写**，嵌入模型本身已内化了改写能力
4. 无需对话-文档相关性标注，只需对话-改写对

关键设计：分布偏移下，改写查询的嵌入比原始对话查询嵌入更稳定 — RCEM 通过对齐将这种稳定性转移给了对话嵌入。

---

## 故事弧 / Story Arc

> *"Conversational retrieval breaks under distributional shift because raw dialogue embeddings are distribution-sensitive. LLM-rewritten queries are more stable, but calling an LLM at inference is expensive. RCEM distills the rewriting capability into the embedding model itself, so the embedder becomes distribution-robust without any inference-time rewriting overhead."*

---

## 创新点 / Innovation

1. **蒸馏至嵌入层**：将 LLM 查询改写能力蒸馏到 embedding model，而非蒸馏到轻量 LM；这是专为密集检索场景设计的
2. **无需相关性标注**：训练只需 (对话查询, 改写查询) 对，比传统监督检索训练成本低得多
3. **分布偏移鲁棒性**：实验专门设计跨分布测试（train-test domain mismatch），这在检索研究中较少被系统探讨

与相关工作比较：
| 方法 | 推理时改写 | 相关性标注依赖 | 分布偏移鲁棒 |
|------|-----------|--------------|------------|
| ConvDR | ❌ | ✅（需要） | 弱 |
| TopiOCQA Baselines | ❌ | ✅ | 弱 |
| Query Rewriting + Retriever | ✅（需LLM推理）| ❌ | 强 |
| **RCEM** | ❌（推理无需改写） | ❌ | **强** |

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | RCEM 提升 |
|--------|------|-----------|
| QReCC (cross-domain) | Recall@10 | **+20% vs strong baselines** |
| TopiOCQA | Recall@10 | consistent outperformance |
| TREC CAsT | Recall@10 | consistent outperformance |

分布偏移场景下提升尤为显著。

---

## 打分明细 / Scoring Breakdown

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | 将LLM改写能力蒸馏至embedding，训练数据需求少，思路清晰 |
| Experimental SOTA Delta | 11 | 15 | Recall@10最高+20%，实用价值高 |
| Experimental Quality/Ablations | 11 | 15 | 3个基准测试；微软来源，实验质量可靠 |
| Efficiency | 8 | 10 | 推理时无需调用LLM改写，延迟大幅降低 |
| Generalization | 4 | 5 | 3个数据集，跨域测试 |
| Domain Relevance | 12 | 25 | 对话式商品搜索场景相关；电商搜索引擎优化有参考价值 |
| **Total** | **66** | **100** | |

---

## 与电商/内容治理的关联

- **对话式商品搜索**：用户多轮询问商品细节时（"有没有带夹克领的？""上次那个颜色有其他款吗？"），RCEM 可在保持上下文感知的同时，不增加推理延迟
- **达人审核对话式查询**：运营人员通过自然语言对话检索历史违规案例时，RCEM 提高召回率并保持对不同表达风格的鲁棒性
- **分布偏移问题**在电商中高发（节日大促 vs 日常、PC 端 vs 移动端搜索风格差异），RCEM 对此有针对性设计
