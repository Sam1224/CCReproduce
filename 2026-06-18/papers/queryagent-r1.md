# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | (Alibaba International Digital Commerce Group) |
| **Affiliation** | Alibaba International Digital Commercial Group |
| **arXiv ID** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | ~June 4, 2026 |
| **Venue** | Preprint / Industry |
| **Code** | Not released (proprietary deployment) |
| **Reproduction** | [`code/QueryAgent-R1/`](../../code/QueryAgent-R1/) |

---

## 方法概述 / Method Summary

### Story Arc

> **现有方法的问题**：电商搜索中的查询推荐系统通常只优化查询点击率（CTR），而忽略了被推荐查询所检索到的商品是否真正符合用户的下游购买意图。这种"查询级"优化与商品级转化率（CVR）之间的鸿沟导致用户体验下降。
>
> **解决方案**：提出QueryAgent-R1，一个记忆增强的智能体框架，通过"检索链"（chain-of-retrieval）优化实现端到端对齐——将查询生成锚定在真实商品库存检索上，让智能体能够根据检索到的商品验证并精化生成的查询，同时在强化学习（RL）过程中设计一致性奖励（consistency reward）来联合优化查询相关性和下游用户参与度。

### Technical Approach (EN)

QueryAgent-R1 is a memory-augmented agentic framework for e-commerce query recommendation that closes the gap between query-level click-through rate (CTR) optimization and product-level conversion rate (CVR) alignment. The key novelties are:

1. **Chain-of-Retrieval Optimization**: Instead of generating queries in isolation, the agent iteratively retrieves actual product results for each candidate query and uses those results to validate or refine the query. This grounds generation in real inventory.
2. **Consistency Reward in Agentic RL**: A new RL reward signal measures whether the products retrieved by a generated query match the user's true downstream purchase intent, creating a joint signal across query generation and product retrieval stages.
3. **Memory Module**: Stores successful historical query-retrieval-purchase chains to condition future generation and reduce exploration cost in RL fine-tuning.

The framework is deployed on a large-scale e-commerce platform with tens of millions of daily active users (Alibaba International).

### 创新亮点 (ZH)

- **端到端RL对齐**：首次将查询生成与商品检索纳入统一强化学习目标，解决了以往"查询CTR高但CVR低"的矛盾。
- **一致性奖励设计**：基于检索结果与用户历史购买行为的一致性构造奖励信号，无需额外标注。
- **记忆增强**：复用历史成功链路降低RL采样代价，提升样本效率。

---

## 关键指标 / Key Metrics

| Dataset | Metric | QueryAgent-R1 | Best Baseline |
|---------|--------|---------------|---------------|
| Proprietary Industrial | CTR (online A/B) | +↑ significant | vanilla query gen |
| Proprietary Industrial | CVR (online A/B) | +↑ significant | vanilla query gen |
| Public E-com Benchmark | NDCG@5 | SOTA | prior RLHF methods |

*Exact absolute numbers are proprietary; paper reports statistically significant online improvements from A/B testing on tens-of-millions-DAU platform.*

---

## 评分详情 / Scoring Breakdown

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation | 25/30 | Novel chain-of-retrieval + consistency reward; genuine RL-grounding of query generation in real inventory |
| Experimental SOTA delta | 12/15 | A/B tested at scale; public benchmark SOTA; absolute numbers proprietary |
| Experimental quality / ablations | 12/15 | Ablations on chain-of-retrieval, memory module, consistency reward shown |
| Efficiency | 8/10 | Production deployed at Alibaba International, tens of millions DAU |
| Generalization | 4/5 | Covers multiple product categories and markets |
| Domain relevance (ecom+governance) | 24/25 | Direct e-commerce application; query generation; agent; RL — core domain |
| **Total** | **85/100** | |

---

## 差异化分析 / Novelty vs. Prior Work

| Prior Work | Gap | QueryAgent-R1 Contribution |
|---|---|---|
| BM25 / BERT query expansion | Static retrieval, no RL | Dynamic RL with retrieval feedback |
| LLM query generation (no retrieval grounding) | Hallucinated products | Grounded in real inventory |
| RLHF for ranking | Item ranking only | End-to-end query→retrieval→CVR |
