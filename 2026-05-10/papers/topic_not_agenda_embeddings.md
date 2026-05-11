## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Topic Is Not Agenda: A Citation-Community Audit of Text Embeddings |
| **arXiv ID** | [2605.07158](https://arxiv.org/abs/2605.07158) |
| **提交日期** | 2026-05（May 2026） |
| **作者** | Junseon Yoo |
| **机构** | 待补充 |
| **领域** | Text Embeddings · RAG Quality · Vector Search · Information Retrieval |
| **Bucket** | WEAK |

---

## 方法概述 / Method Summary

本文通过大规模引用图审计，揭示文本嵌入（Text Embeddings）在向量搜索和 RAG 中的根本性局限：

**核心发现：** 余弦相似度 ≈ 概念相关性（这一 RAG 的基础假设）在"研究议程"层级严重失效。

**方法：**
1. 构建 358 万篇科学论文的增强引用图
2. 用 Leiden CPM 算法在两个粒度划分社区：
   - L1（子领域级）：粗粒度话题分组
   - L2（研究议程级）：精细化研究方向
3. 评测 4 个 SOTA 嵌入模型（Gemini、Qwen3-8B、Qwen3-0.6B、SPECTER2）

**结果：**
- L1 级别：45-52% 的 top-10 邻居来自同一子领域（尚可）
- L2 级别：只有 15-21% 的 top-10 邻居共享同一研究议程
- **即：10 篇检索结果中，8 篇与查询的研究议程不一致**

### Story Arc
> **RAG 和向量搜索依赖"余弦相似度 = 概念相关"这一假设** → 在 358 万篇科学论文的严格引用社区测试中，4 个顶级嵌入模型在"研究议程"精度上均只有 15-21%，80% 的检索结果是"议程外"的——这一缺陷在需要精准领域对齐的电商搜索（如特定品类/违规类型精准匹配）中危害极大。

---

## 关键指标 / Key Metrics

| 模型 | L1 同社区 Top-10 率 | L2 同议程 Top-10 率 |
|------|------------------|------------------|
| Gemini Embedding | ~52% | ~21% |
| Qwen3-8B | ~50% | ~20% |
| Qwen3-0.6B | ~47% | ~17% |
| SPECTER2 | ~45% | ~15% |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 14/30 | 重要发现但分析框架非全新 |
| SOTA Delta | 8/15 | 诊断性研究，无改进 |
| Exp Quality / Ablations | 10/15 | 358 万论文的规模，4 模型对比 |
| Efficiency | 8/10 | 分析框架轻量 |
| Generalization | 3/5 | 科学文献域，向电商泛化需工作 |
| Domain Relevance | 11/25 | RAG 质量风险警示，对电商检索有参考价值 |
| **总分** | **54/100** | Feishu 推送 |
