# LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG |
| **arXiv ID** | [2605.06285](https://arxiv.org/abs/2605.06285) |
| **提交日期** | 2026-05-08（约） |
| **作者** | 未完全披露 |
| **机构** | 未完全披露 |
| **领域桶** | WEAK |
| **综合评分** | **78 / 100** |

---

## 方法概述 (Chinese)

传统 RAG 的单步检索难以处理复杂问题；Agentic RAG 通过 LLM 多步推理生成中间思路和子查询，但逐 token 自回归生成带来显著的推理延迟。

LatentRAG 将推理和检索从**离散语言空间**迁移至**连续隐空间（latent space）**：
- LLM 不再生成自然语言形式的"思路"和"子查询"，而是从隐状态直接生成**隐向量 tokens**；
- 通过隐空间将 LLM 与密集检索模型对齐，支持对隐子查询 tokens 的直接检索；
- 支持端到端联合优化（end-to-end joint optimization）。

在 7 个基准数据集上，LatentRAG 的准确率与显式 Agentic RAG 方法相当，同时将推理延迟降低约 **90%**，大幅缩小与传统单步 RAG 的延迟差距。

## Method Overview (English)

LatentRAG shifts both reasoning and retrieval from discrete language space to continuous latent space. Instead of generating natural language thoughts and subqueries token-by-token, the LLM produces latent tokens for both in a single forward pass. The LLM and dense retriever are aligned in latent space for end-to-end joint optimization. Results across 7 benchmarks: comparable accuracy to explicit agentic RAG at ~90% lower inference latency.

---

## Story Arc

**Agentic RAG 通过自回归生成中间思路和子查询解决了复杂问题推理，但高延迟制约了实际部署 → LatentRAG 将推理和检索迁移至连续隐空间，单次前向传递生成隐向量思路和子查询，以90%更低延迟达到相当准确率。**

> Agentic RAG thinks out loud in English — verbose and slow. LatentRAG thinks in "thought vectors" — compact, fast, and equally accurate.

---

## 创新性分析

1. **隐空间推理范式**：将 LLM 推理迁移至连续空间是根本性创新，突破了自回归延迟瓶颈；
2. **LLM-检索器隐空间对齐**：两个异构模型在隐空间的联合优化是技术难点突破；
3. **单次前向传递**：消除了多次自回归推理调用，90%延迟降低具有实际部署价值；
4. **端到端优化**：允许检索和推理共同优化，而非流水线独立优化。

**与先前工作的差异**：ReAct、IRCoT、FLARE 等 Agentic RAG 方法均在语言空间操作；LatentRAG 是首个在隐空间完成多步 RAG 推理的方法（据其声明）。

**与电商内容生态的关联**：高效 RAG 可应用于电商产品知识库检索、内容审核规则查询、达人内容质量评估等场景。

---

## 关键指标 / Key Metrics

| 基准数据集 | 精度 vs. 显式AgRAG | 延迟降低 |
|---|---|---|
| 7个多跳QA基准（综合） | 相当（无显著损失） | **~90%** |
| 对比单步RAG | 准确率更高 | 延迟接近单步 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 24 | 隐空间推理范式是根本性创新，突破自回归延迟瓶颈 |
| 实验SOTA增益 (SOTA delta) | 15 | 12 | 7基准准确率持平+90%延迟降低 |
| 实验质量与消融 (Quality) | 15 | 12 | 7基准数据集，端到端优化验证 |
| 效率 (Efficiency) | 10 | 10 | 90%延迟降低是极强效率贡献 |
| 泛化性 (Generalization) | 5 | 5 | 7个多样化基准 |
| 领域相关性 (Domain) | 25 | 15 | RAG可迁移至电商知识检索，但非直接电商/内容治理 |
| **总分** | **100** | **78** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.06285
- HTML版: https://arxiv.org/html/2605.06285
