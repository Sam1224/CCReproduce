# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| Field | Detail |
|-------|--------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| **Affiliation** | Alibaba International Digital Commercial Group (AIDC) |
| **arXiv ID** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | 4 June 2026 (indexed in 8 June 2026 listing) |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

**问题背景 / Problem**: 电商搜索中的查询推荐（Query Recommendation）旨在主动为用户推荐符合潜在兴趣的搜索词。现有方法主要优化**查询级相关性**，但忽视了推荐查询召回的商品是否符合用户下游购买意图。这导致一个普遍现象：**高点击率（CTR）但低转化率（CVR）**——用户点了查询但未找到心仪商品。

**Story Arc**: *"优化点击忽视转化，CTR 与 CVR 脱节 → 设计 QueryAgent-R1，通过实时商品检索验证查询质量，RL 训练一致性奖励打通端到端对齐"*

**方法 / Method**:

QueryAgent-R1 是一个记忆增强（Memory-Augmented）的 Agentic RL 框架，核心三个组件：

1. **Chain-of-Retrieval (CoR) Optimization**:
   - Agent 生成候选查询 → 调用真实商品检索引擎 → 观察检索结果
   - 基于检索结果验证并精炼查询
   - RL 训练信号来自最终商品匹配质量（而非单纯查询文本匹配）

2. **Consistency Reward**:
   - 联合优化：查询相关性（Query CTR）+ 商品匹配度（Product CVR）
   - 奖励函数结合用户点击与购买信号，构建端到端对齐

3. **Memory Abstraction (兴趣图)**:
   - 从用户长期行为历史中提取 Interest Graph
   - 高效编码用户多维偏好，用于个性化查询生成

**架构图**:
```
User Long-term History
        │
        ▼
┌──────────────────┐
│  Memory Module   │ ← Interest Graph
│  (兴趣图抽取)     │
└──────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  QueryAgent-R1 (LLM-based Agent)        │
│  ─────────────────────────────────────  │
│  Step 1: Generate candidate queries     │
│  Step 2: Retrieve products (real index) │ ← Chain-of-Retrieval
│  Step 3: Validate & refine queries      │
└──────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────┐
│  Consistency Reward (RL)   │
│  = w1*CTR + w2*CVR_signal  │
└────────────────────────────┘
        │
        ▼
  Final Query Recommendations
```

---

## 故事弧 / Story Arc

- **Insufficient**: 现有查询推荐模型只看查询文本相关性，实际落地 CTR 高但用户最终找不到想买的商品
- **We Design**: QueryAgent-R1，在查询生成过程中嵌入真实商品检索，以 RL 一致性奖励端到端对齐查询点击与商品转化
- **Result**: 在 AIDC 平台（数千万 DAU）线上部署，显著提升 CTR-CVR 联合指标

---

## 创新分析 / Innovation

| 维度 | 分析 |
|------|------|
| **Chain-of-Retrieval** | 将商品检索融入查询生成的推理链，真正闭合 query→product 反馈环路 |
| **Consistency Reward** | 同时优化两个下游目标，避免 CTR-only 优化的局部最优 |
| **Memory Abstraction** | Interest Graph 提供轻量级用户模型，避免直接处理原始交互序列 |
| **RL 框架** | 类 R1/GRPO 风格的 agentic RL，将商品检索 API 作为 tool |
| **可行性** | 已上线，提交时给出 offline + online 实验 |

---

## 关键指标 / Key Metrics

| Dataset / Platform | Metric | QueryAgent-R1 | Baseline |
|-------------------|--------|---------------|---------|
| AIDC Proprietary Dataset | Offline Metrics | Improved (specific values in paper) | Existing methods |
| AIDC Public Dataset | Offline Metrics | Improved | Existing methods |
| AIDC Production (tens-of-millions DAU) | CTR + CVR | Joint improvement | Baseline system |

*注：具体数值需见全文；该框架已通过大规模 A/B 测试验证*

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 24 | CoR + Consistency Reward 组合创新，R1 style RL 用于查询推荐属新颖 |
| SOTA Delta | 15 | 9 | 大规模上线验证，但搜索到的精确指标有限 |
| Experimental Quality | 15 | 10 | 两个数据集 + 线上实验，较扎实 |
| Efficiency | 10 | 7 | Memory Graph 减少序列处理开销 |
| Generalization | 5 | 4 | 应用于大规模电商平台，泛化能力强 |
| Domain Relevance | 25 | 23 | 直接解决电商查询推荐核心问题，高度相关 |
| **Total** | **100** | **77** | 强候选，电商搜索核心场景 |
