# ExpWeaver: LLM Agents Learn from Experience via Latent RAG

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | ExpWeaver: LLM Agents Learn from Experience via Latent RAG |
| **Authors** | Tao Feng, Tianyang Luo, Jingjun Xu, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You |
| **Affiliations** | (approx.) Multiple institutions; Jiaxuan You is known for UIUC/industry work |
| **Link** | https://arxiv.org/abs/2606.01041 |
| **Submission Date** | 2026-05-31 (appeared in 2026-06-02 listing) |
| **Domain Bucket** | **WEAK** — novel agent memory architecture; transferable to product-search agents, recommendation agents |
| **Total** | **72 / 100** |

---

## 方法概述 / Method Overview

### 问题背景 (Problem)
现有基于经验的 LLM Agent 方法将历史交互作为**显式文本**存储，检索时通过语义相似度召回并拼接到上下文窗口中。这带来两大问题：
1. **Token 开销巨大**：大量经验文本占据上下文，挤压推理空间
2. **架构解耦**：检索模块与生成模块相互独立，无法端到端优化，检索质量上限受限

> X is insufficient: Text-based RAG for agent experience learning creates token overhead and a decoupled architecture that cannot be jointly optimized.

### 设计 (Design — ExpWeaver)
ExpWeaver 将经验学习转移至**潜在空间（Latent Space）**：

1. **经验编码**：用 LLM 自身的隐状态（hidden states）编码历史经验，形成经验向量库
2. **潜在检索**：在**每个解码步**，以当前 hidden state 为查询，对经验库进行向量相似度检索
3. **跨注意力聚合**：通过 cross-attention（带可学习查询 token）将检索到的经验聚合
4. **门控残差融合**：Gated residual mechanism 将聚合结果注入主干模型
5. **端到端 RL 训练**：整条 pipeline 用强化学习端到端优化，同时支持生成任务和排序任务

架构亮点：**无需独立 RAG 模块**，检索发生在 LLM 内部的潜在空间中，解码效率更高。

---

## 故事弧 / Story Arc

> *"Experience utilization for LLM agents has been confined to explicit text space — retrieved by semantic similarity, concatenated into context. This imposes token overhead and prevents joint optimization. ExpWeaver moves both storage and retrieval into the LLM's own latent space, enabling end-to-end reinforcement learning of the entire experience-retrieve-generate pipeline."*

---

## 创新点 / Innovation

1. **潜在空间经验检索（Latent-Space RAG）**：全球首次在 LLM 内部 hidden state 层面做经验检索与融合，区别于 RETRO、Memorizing Transformer 等外部检索方法
2. **逐解码步检索**：传统方法在生成前一次性检索；ExpWeaver 在每个解码步动态更新检索结果，实现细粒度经验影响
3. **端到端 RL 优化**：检索与生成联合训练，避免了两阶段方法的误差累积
4. **无额外参数架构**：无独立检索模块，参数效率高

与相关工作比较：
| 方法 | 检索空间 | 检索时机 | 联合优化 |
|------|----------|----------|---------|
| ExpeL / Self-RAG | 显式文本 | 生成前 | ❌ |
| RETRO | 显式文本/数据库 | 生成前 | 部分 |
| **ExpWeaver** | LLM 潜在空间 | 每个解码步 | ✅（端到端RL）|

---

## 关键指标 / Key Metrics

| 基准 | 任务类型 | ExpWeaver vs. 基线 |
|------|----------|-------------------|
| 13 diverse tasks | QA, reasoning, coding, recommendation, scientific prediction | SOTA on 12/13 tasks |
| Average | All tasks | **>6.8%** over best baseline |

---

## 打分明细 / Scoring Breakdown

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 25 | 30 | 潜在空间经验检索 + 逐解码步融合 + 端到端RL，架构新颖性高 |
| Experimental SOTA Delta | 12 | 15 | 12/13任务SOTA，>6.8%提升，任务覆盖广 |
| Experimental Quality/Ablations | 11 | 15 | 13任务多样，但详细消融信息有限 |
| Efficiency | 8 | 10 | 消除独立RAG模块，减少token开销 |
| Generalization | 4 | 5 | 涵盖QA/推理/编程/推荐等13类任务 |
| Domain Relevance | 12 | 25 | 与产品搜索agent、推荐系统agent相关，但非直接电商/治理应用 |
| **Total** | **72** | **100** | |

---

## 与电商/内容治理的关联

- **商品搜索 Agent**：用户反复搜索的历史经验（"什么表达式最能召回羽绒服"）可存入潜在经验库，无需重新推理
- **内容审核 Agent**：审核决策历史作为经验，新内容到达时动态融合历史判例，提升一致性
- **达人管理流水线**：将过往治理案例编码为 latent experience，处理新案例时自动参考，提升判断效率
