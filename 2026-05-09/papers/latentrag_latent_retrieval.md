# LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG |
| 作者 | 未完全披露 |
| 机构 | 未完全披露 |
| arXiv | https://arxiv.org/abs/2605.06285 |
| 提交日期 | 2026-05-07 |
| 领域标签 | RAG · 隐空间推理 · 检索增强生成 · Agent · 效率优化 |
| 桶类别 | WEAK |
| 综合评分 | **73 / 100** |

---

## 方法概述 (中文)

Agentic RAG 通过多步骤迭代检索解决复杂问题，但每轮迭代都需要 LLM 自回归生成中间思路和子查询，导致推理延迟极高（比单步 RAG 慢 10× 以上），严重制约在线应用。

**LatentRAG** 将推理和检索从离散语言空间整体迁移至**连续隐变量空间**：
1. **隐思路（Latent Thoughts）**：LLM 不再自回归生成文本形式的中间推理步骤，而是从隐藏层直接产生连续隐状态，代表中间推理。
2. **隐子查询（Latent Subqueries）**：同样在隐空间生成检索子查询向量，与密集检索模型（Dense Retriever）在同一嵌入空间对齐，实现直接检索。
3. **单次前向传播**：整个多步推理+检索流程在一次前向传播中完成，无需多轮自回归循环。
4. **并行隐解码（Parallel Latent Decoding）**：将隐状态映射回自然语言，提升透明度并鼓励语义有意义的隐表示。
5. **端到端联合优化**：LLM 与检索模型在隐空间中联合优化，实现协同对齐。

---

## 故事线 (Story Arc)

> **现状不足：** Agentic RAG 的多步检索推理链每轮都需要自回归生成长文本，推理延迟比单步 RAG 高 10× 以上，无法满足电商搜索/推荐等在线场景对毫秒级响应的要求。
>
> **我们的解法：** 将推理和检索全部迁移到连续隐空间，单次前向传播完成多步推理+检索，推理延迟降低 ~90%，同时保持与显式 Agentic RAG 相当的回答质量——LatentRAG。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 全隐空间多步推理+检索，无需自回归文本生成 |
| vs. 先前工作 | 现有 Agentic RAG（如 IRCoT、Self-Ask）依赖显式文本生成中间步骤；LatentRAG 消除该瓶颈 |
| 效率突破 | ~90% 延迟降低是工业级突破，接近单步 RAG 延迟 |
| 可解释性 | 并行隐解码保留了中间推理的文本可读性 |
| 局限 | 隐空间对齐训练成本较高；在领域外数据上的迁移性未深入验证 |

---

## 关键指标

| 数据集 | 指标 | LatentRAG | Agentic RAG 基线 |
|--------|------|-----------|----------------|
| 7 个多跳 QA 基准（含 HotpotQA, MuSiQue等） | EM / F1 | 与显式 Agentic RAG **相当** | IRCoT, Self-Ask |
| 推理延迟 | ms | **~90% 降低** | Agentic RAG 基线 |
| vs. 单步 RAG | 延迟差距 | **大幅收窄** | 单步 RAG |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 23 | 30 | 全隐空间多步RAG架构创新显著 |
| Experimental SOTA delta | 12 | 15 | 90%延迟降低+7基准相当精度 |
| Experimental quality | 12 | 15 | 7个多跳QA基准，消融实验充分 |
| Efficiency | 10 | 10 | 90%延迟降低，工业价值极高 |
| Generalization | 4 | 5 | 7个多样QA基准验证 |
| Domain relevance | 12 | 25 | 通用RAG，可用于电商问答/商品检索知识增强 |
| **Total** | **73** | **100** | |
