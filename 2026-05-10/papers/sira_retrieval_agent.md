## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Superintelligent Retrieval Agent: The Next Frontier of Information Retrieval |
| **arXiv ID** | [2605.06647](https://arxiv.org/abs/2605.06647) |
| **提交日期** | 2026-05-07 |
| **作者** | Zeyu Yang, Qi Ma, Jason Chen, Anshumali Shrivastava |
| **机构** | Meta AI Research · Rice University |
| **代码** | https://github.com/facebookresearch/sira |
| **领域** | Information Retrieval · RAG · LLM Agent · BM25 |
| **Bucket** | WEAK |

---

## 方法概述 / Method Summary

SIRA（Superintelligent Retrieval Agent）将"超级智能检索"定义为：将多轮探索性搜索压缩为单次语料库判别式检索动作的能力。

**核心思想：** LLM 对检索语料库有隐式知识，可预判"哪些术语能将目标文档与语料库混淆项区分开来"，而非仅分析"哪些术语与查询相关"。

**具体流程：**
1. **语料库侧离线增强**：LLM 对每个文档离线补充缺失的搜索词汇（例如：同义词、缩写、相关概念）
2. **查询侧预测**：LLM 预测目标文档中查询者可能漏写的证据词汇
3. **工具调用过滤**：用文档频率统计工具调用过滤掉缺失、过于常见或无法区分的候选词
4. **单次 BM25 检索**：将原始查询与验证后的扩展词汇组合，执行一次加权 BM25 调用

**对比传统方法：** 传统 agentic 检索需多轮循环查询；SIRA 单次调用即超越多轮方法，且完全可解释。

### Story Arc
> **检索增强 Agent 通常像"新手"一样多轮摸索未知数据库** → SIRA 赋予 LLM 像"专家"一样预判检索语料库术语分布，通过语料库-查询双侧词汇增强，将多轮搜索压缩为单次 BM25 调用，在 BEIR 10 项基准上超越密集检索和多轮 agentic 方法。

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | SIRA | Dense Retrieval | Multi-Round Agent |
|--------|------|------|----------------|-------------------|
| BEIR (10 benchmarks avg) | nDCG@10 | **SOTA** | 劣于 SIRA | 劣于 SIRA |
| QA Downstream Tasks | Accuracy | 显著提升 | — | — |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 16/30 | 语料库+查询双侧增强+判别性词汇预测，思路独特 |
| SOTA Delta | 10/15 | BEIR 10 项全面超越，实验充分 |
| Exp Quality / Ablations | 10/15 | 多基准全面测试，提供代码 |
| Efficiency | 9/10 | 单次 BM25，无需训练，极低延迟 |
| Generalization | 4/5 | 10 个 BEIR 基准覆盖广 |
| Domain Relevance | 12/25 | 通用 IR，适用于电商搜索召回层 |
| **总分** | **61/100** | Feishu 推送 |
