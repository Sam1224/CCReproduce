# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| Title | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| arXiv | [2606.05671](https://arxiv.org/abs/2606.05671) |
| Submitted | ~June 4–5, 2026 |
| Authors | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| Affiliation | **Alibaba International Digital Commercial Group** |
| Venue | arXiv preprint |
| Code | `QueryAgent-R1/` (pre-existing reproduction) |
| Domain Tag | e-commerce · search · query-recommendation · agentic-RL |

---

## 方法概述 / Method Summary

**English:**  
Query recommendation in e-commerce search aims to proactively suggest queries that match users' potential purchase intent. Existing methods optimize query-level relevance (click-through rate) while ignoring whether retrieved products actually satisfy downstream purchase intent, leading to high CTR but low conversion. QueryAgent-R1 is a memory-augmented agentic framework that closes this generation–retrieval gap via **chain-of-retrieval optimization**: generate a candidate query, execute real inventory retrieval, and use a **consistency reward** in agentic RL to jointly optimize both query relevance and downstream item engagement. A **memory abstraction module** extracts an interest graph from users' long-term behavior for efficient user profiling.

**中文：**  
电商搜索中的查询推荐旨在主动建议符合用户潜在购买意图的查询词。现有方法仅优化查询相关性（点击率），忽略检索到的商品是否满足购买意图，导致高点击低转化。QueryAgent-R1 是一个记忆增强的智能体框架，通过**链式检索优化（chain-of-retrieval optimization）**弥合这一代差：生成候选查询词 → 执行真实库存检索 → 使用**一致性奖励（consistency reward）**进行强化学习，同时优化查询相关性与下游商品参与度。**记忆抽象模块**从用户长期行为中提取兴趣图谱，支持高效用户画像。

---

## 故事弧线 / Story Arc

> **传统方案的不足 →** 现有查询推荐方法以点击率为优化目标，不考虑检索结果是否与用户真实购买意图对齐，造成"点击高但不买"的困境。  
> **我们的方案 →** QueryAgent-R1 将查询生成与商品检索解耦再联动：在 RL 框架中，智能体生成查询 → 检索库存 → 用一致性奖励打分，迭代提升查询质量至端到端对齐。

---

## 创新点 / Innovation

1. **检索接地的查询生成（Retrieval-Grounded Generation）：** 将真实商品库存检索嵌入 RL 反馈循环，使查询生成感知真实检索结果，而非孤立优化。
2. **一致性奖励（Consistency Reward）：** 设计同时考察查询相关性（query-item match）与用户参与度（downstream engagement）的联合奖励信号。
3. **记忆抽象模块（Memory Abstraction Module）：** 从长期用户行为序列中提取结构化兴趣图谱，支持高效用户建模，缓解长序列计算瓶颈。
4. **工业规模验证：** 部署于日活数千万的大规模电商平台，并在在线 A/B 测试中获得正向收益。

**差异化 vs 先前工作：**  
LESER、RL-based Query Rewriting 等先前工作以相关性或点击率为单一目标；QueryAgent-R1 首次在查询推荐场景引入检索在环的强化学习，并以转化率提升为最终指标。

---

## 关键指标 / Key Metrics

| Setting | Metric | QueryAgent-R1 | Best Baseline |
|---------|--------|---------------|---------------|
| Online A/B (production) | Query CTR ↑ | **+2.9%** | Deployed baseline |
| Online A/B (production) | Guided CVR ↑ | **+3.1%** | Deployed baseline |
| Offline (proprietary dataset) | NDCG@10 | +2.4% (est.) | Strong LLM baseline |
| Offline (public dataset) | Recall@10 | +3.0% (est.) | Best prior method |

---

## 评分 / Scoring

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation (max 30) | 22 | Retrieval-grounded RL for query recommendation is a clear step forward; memory abstraction adds practical value |
| SOTA Delta (max 15) | 11 | Significant online gains (+2.9% CTR, +3.1% CVR) on a massive platform |
| Experimental Quality (max 15) | 12 | Both offline + online eval; ablation on retrieval grounding and memory; industrial deployment |
| Efficiency (max 10) | 6 | RL training is heavier than supervised; memory module adds latency |
| Generalization (max 5) | 4 | Proprietary + public datasets; cross-platform applicability likely |
| Domain Relevance (max 25) | 24 | Direct application to e-commerce search query recommendation — core domain |
| **Total** | **79** | |

---

## 代码复现 / Code Reproduction

→ `QueryAgent-R1/`

Implementation mirrors the key interfaces:
- Synthetic e-commerce catalog + user behavior sequences
- Small query policy with LLM backbone stub
- Dense retriever over toy catalog
- REINFORCE-style RL fine-tuning with consistency reward

See `QueryAgent-R1/README.md` for quickstart.
