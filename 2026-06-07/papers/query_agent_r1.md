## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| **Affiliations** | Alibaba International Digital Commercial Group |
| **ArXiv ID** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | ~2026-06-04 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.IR, cs.AI |
| **Code** | No official code repo found |
| **Reproduction** | `code/QueryAgentR1/` |
| **Bucket** | STRONG |
| **Total** | **82 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 23 | 30 | Chain-of-retrieval optimization grounding query generation in real inventory is genuinely novel; consistency reward bridging CTR↔CVR gap is distinct |
| Experimental SOTA delta | 13 | 15 | Online A/B: CTR +2.9%, guided CVR +3.1%; outperforms strong baselines on offline datasets |
| Experimental quality / ablations | 12 | 15 | Multiple offline datasets + real online A/B; ablates memory module and reward design |
| Efficiency | 7 | 10 | Memory abstraction (interest graph) amortizes long-term user profiling cost; adds retrieval step latency |
| Generalization | 3 | 5 | E-commerce specific; architecture is transferable to any inventory-grounded agent |
| Domain relevance | 24 | 25 | Directly solves e-commerce search query recommendation — a core metric driver (CTR + CVR alignment) |
| **Total** | **82** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
电商查询推荐（Query Recommendation）的现有方法主要优化"查询级相关性"（高 CTR），却忽视最终检索到的商品是否真正符合用户的转化意图（CVR），导致"点击多但下单少"的困境。

**X is insufficient → we design Y to solve it：**
> 现有方法仅优化 query 与用户兴趣的语义相关性，忽略了"用户点击推荐 query 后，能否在实际商品库中找到满足其偏好的商品"。QueryAgent-R1 设计了"检索接地"的 agentic RL 框架，让 agent 生成 query → 实际执行检索 → 基于检索结果反思并精炼 query，形成 Chain-of-Retrieval 的端到端优化循环。

### 核心方法

1. **记忆抽象模块（Memory Abstraction）**：从用户长期交互历史中提取兴趣图（Interest Graph），压缩为高效的用户画像 embedding，避免暴力堆叠长序列。

2. **Chain-of-Retrieval 优化**：agent 迭代地：(a) 根据用户记忆生成候选 query；(b) 用该 query 在真实商品库中执行检索；(c) 根据检索结果判断 query 是否对齐用户偏好；(d) 必要时重写 query。

3. **Consistency Reward（一致性奖励）**：在 RL 训练中设计联合奖励函数，同时优化 query 相关性（CTR 代理信号）和下游商品参与度（CVR 代理信号），使两个目标对齐而非对立。

4. **在线 A/B 测试**：在阿里国际电商平台实际流量上验证效果。

### English Summary

QueryAgent-R1 is a memory-augmented agentic framework for e-commerce query recommendation that closes the query-product alignment gap. It introduces a Chain-of-Retrieval loop: the agent generates a query, executes it against the real product inventory, observes what products are retrieved, and refines the query based on that feedback. A consistency reward in the RL training objective jointly optimizes query click-through rate (CTR) and guided conversion rate (CVR), eliminating the classic tension between relevance and conversion. A memory abstraction module compresses long-term user interaction history into a compact interest graph, enabling efficient personalization at scale.

---

## 创新点分析 / Innovation Analysis

**中文：** 核心创新在于将"真实商品库检索"引入 query 推荐的训练环路——这是首个把"检索结果"作为强化学习奖励信号的一部分来联合优化 query 生成的工作（据论文声明）。与 GRPO/PPO 直接用文本奖励的 query 生成方法相比，QueryAgent-R1 的 consistency reward 考虑了"检索执行后商品集合是否满足用户意图"，形成更对齐实际业务目标的优化信号。记忆模块（Interest Graph）的引入也让系统在实际大规模部署中可行。

**English:** The core novelty is injecting real inventory retrieval into the training loop as part of the RL reward signal for query generation — the agent's reward depends not just on query-user similarity but on whether the retrieved product set satisfies user intent. This grounds abstract language-space optimization in concrete business-metric-aligned feedback. The Interest Graph memory module is a pragmatic solution to efficient long-term user profiling at industrial scale.

**Plausibility / Feasibility:** High. The architecture is a clean composition of standard components (LLM generator + retrieval API + RL fine-tuning), and online A/B results demonstrate real-world efficacy.

---

## 关键指标 / Key Metrics

| Dataset / Setting | Metric | Value | Baseline |
|-------------------|--------|-------|----------|
| Online A/B (Alibaba Intl) | Query CTR | **+2.9%** | Production system |
| Online A/B (Alibaba Intl) | Guided CVR | **+3.1%** | Production system |
| Offline benchmarks (multiple) | Outperforms strong baselines | ✓ | Multiple baselines |

---

## 代码复现 / Code Reproduction

官方代码未发布。本仓库提供基于论文的 PyTorch 复现实现：
→ **`code/QueryAgentR1/`**

复现内容：
- `model.py` — Interest Graph 记忆模块 + LLM-based Query Generator
- `reward.py` — Consistency Reward 计算（query relevance + retrieval coverage）
- `train.py` — GRPO/PPO RL 训练脚本（toy 规模）
- `retrieval.py` — 商品检索接口（BM25 toy 版）
- `data.py` — 数据接口（toy e-commerce query-product pairs）
- `eval.py` — 离线评估脚本
