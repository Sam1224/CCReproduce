# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| **Affiliation** | Alibaba International Digital Commercial Group |
| **arXiv** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | June 4, 2026 (indexed June 5–6, 2026 GMT+8) |
| **Code** | `code/QueryAgent-R1/` (reproduced; official code not released at time of report) |
| **Bucket** | STRONG |

---

## 故事弧线 / Story Arc

> **"现有查询推荐方法只优化查询级相关性，忽视下游商品是否匹配用户偏好，导致高CTR但低CVR → 我们设计 QueryAgent-R1，通过链式检索优化（chain-of-retrieval）将查询生成与真实库存检索打通，并设计一致性奖励同时提升CTR和CVR。"**

In e-commerce search, query recommendation (suggesting queries to users) typically maximizes click-through rate (CTR) but ignores whether the products retrieved by those queries actually convert. A query can be highly clickable yet surface irrelevant products, hurting conversion rate (CVR). QueryAgent-R1 closes this gap by grounding query generation in real inventory retrieval — the agent generates candidate queries, retrieves actual product results, and evaluates whether those products align with user intent before finalizing the recommendation.

---

## 方法 / Method

**Framework Architecture**:

```
User Profile + Context
        │
        ▼
  Memory Module
  (historical queries, interactions, user preference profile)
        │
        ▼
  Query Generator (LLM backbone)
  → generates K candidate queries
        │
        ▼
  Product Retrieval Engine
  → retrieves top-N products per query from live inventory
        │
        ▼
  Consistency Evaluator
  → measures alignment between retrieved products and user preference
        │
        ▼
  RL Training with Consistency Reward
  reward = f(query CTR estimate, product-user alignment score)
        │
        ▼
  Refined Query Recommendation
```

**Chain-of-Retrieval Optimization**:
- Each RL step includes a real retrieval call against the inventory index
- The agent learns to generate queries whose **retrieved products** maximize user preference alignment, not just click probability

**Reward Design**:
- CTR reward: estimated click probability from historical data
- CVR consistency reward: cosine similarity between retrieved product embeddings and user preference vector
- Joint reward: `r = α·CTR_reward + (1-α)·CVR_consistency_reward`

**Memory Module**:
- Stores user's historical queries and product interactions
- Provides context window for query generation, allowing personalized recommendations

---

## 创新性 / Innovation

| Aspect | Prior Work | QueryAgent-R1 |
|--------|------------|---------------|
| Optimization target | Query-level CTR only | Joint CTR + CVR via product-grounded reward |
| Retrieval loop | Offline / separate | Online chain-of-retrieval in RL loop |
| User modeling | Static embeddings | Memory-augmented user profile |
| Reward signal | Click labels | Consistency between retrieved products and preferences |

**Novelty assessment**: Solid practical contribution. The key insight — that query quality should be evaluated by the **products it retrieves**, not just by click probability — is well-motivated and underexplored. The chain-of-retrieval RL framing elegantly closes the train/deploy gap where retrieval happens only at inference.

---

## 关键指标 / Key Metrics

| Dataset/Context | Metric | QueryAgent-R1 | Baseline |
|----------------|--------|---------------|---------|
| Proprietary industrial dataset | Offline NDCG | **+XX%** (state-of-the-art vs strong baselines) | prior SFT models |
| Public benchmark | Offline Relevance | **consistent improvement** | multiple retrieval baselines |
| Alibaba production (A/B test) | Query CTR | **+2.9%** | control |
| Alibaba production (A/B test) | Guided CVR | **+3.1%** | control |

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 22 | Chain-of-retrieval RL for query recommendation is novel; joint CTR+CVR reward is well-motivated but technically incremental |
| Experimental SOTA delta | 15 | 12 | Production A/B: +2.9% CTR, +3.1% CVR at Alibaba scale is economically significant |
| Experimental quality / ablations | 15 | 11 | Both offline (proprietary + public) and online A/B evaluation; ablations on reward components expected |
| Efficiency | 10 | 7 | Retrieval in RL loop adds latency; memory module overhead; inference at serving scale manageable |
| Generalization | 5 | 3 | Tested primarily on Alibaba International platform; generalization to other e-commerce platforms not demonstrated |
| Domain relevance (ecom + governance) | 25 | 25 | Exact e-commerce query recommendation from Alibaba; direct production deployment |
| **Total** | **100** | **80** | Strong industrial e-commerce agent paper with real A/B validation; moderate novelty |

---

## 复现说明 / Reproduction Notes

Full PyTorch implementation at `code/QueryAgent-R1/`. Covers:
- `model/query_agent.py` — LLM-based query generator with memory module
- `retrieval/retriever.py` — product retrieval engine (toy index)
- `rl/reward.py` — CTR + CVR consistency reward
- `rl/trainer.py` — PPO-based RL training loop
- `data/dataset.py` — e-commerce query/product toy dataset
- `eval.py` — offline NDCG + online CTR/CVR simulation
