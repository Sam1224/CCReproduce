# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **Authors** | (Alibaba International Digital Commerce Group) |
| **Affiliation** | Alibaba International Digital Commerce Group |
| **ArXiv** | [2606.05671](https://arxiv.org/abs/2606.05671) |
| **Submitted** | June 4, 2026 |
| **Domain Tags** | `e-commerce` `recommendation` `RL` `RAG` `agents` `query-generation` |
| **Code** | `code/QueryAgent-R1/` (reproduction) |
| **Total** | **82 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 电商搜索中的查询推荐方法主要优化查询层面的相关性，却忽略了被检索商品是否真正匹配用户的下游转化意图，导致查询点击率（CTR）高但商品转化率（CVR）低的困境。

**解决方案：** QueryAgent-R1 提出以真实商品库检索为基础的链式检索优化（Chain-of-Retrieval Optimization），通过强化学习（RL）一致性奖励联合优化查询相关性与下游商品转化，并引入记忆抽象模块高效管理用户长期兴趣。

**Existing methods are insufficient → QueryAgent-R1 solves it by:**  
现有方法独立优化查询相关性 → QueryAgent-R1 通过将商品检索结果接入 RL 循环，使 Agent 能验证和精化查询，实现端到端 CTR-CVR 对齐。

---

## 方法概述 / Method

QueryAgent-R1 是一个**记忆增强的 Agentic 框架**，通过链式检索优化（Chain-of-Retrieval Optimization）提升端到端对齐：

**核心组件：**

1. **Chain-of-Retrieval（链式检索）**：查询生成 Agent 生成候选查询后，立即调用真实商品检索系统获取检索结果。Agent 根据检索到的商品验证查询质量，并进行精化迭代。这一过程在强化学习训练中内嵌。

2. **Consistency Reward（一致性奖励）**：设计双目标奖励函数，同时优化：
   - **查询相关性奖励**：生成的查询与用户意图的语义匹配度
   - **商品对齐奖励**：检索到的商品与用户历史行为偏好的一致性
   
   两者通过加权求和形成一致性奖励信号，驱动 RL 训练。

3. **Memory Abstraction Module（记忆抽象模块）**：从用户长期行为历史中提取兴趣图谱（interest graph），高效压缩用户画像，避免长上下文导致的计算瓶颈。

**训练方式：** 基于 GRPO（Group Relative Policy Optimization）的强化学习，在 Agent 框架中端到端训练。

---

## 创新性分析 / Innovation

**与现有工作的区别：**

| Aspect | Prior Work | QueryAgent-R1 |
|--------|-----------|---------------|
| 优化目标 | 查询-用户相关性 | CTR + CVR 联合优化 |
| 检索耦合 | 离线评估 | 在线检索嵌入 RL 循环 |
| 用户画像 | 固定嵌入 / 历史序列 | 兴趣图谱（动态更新）|
| 训练范式 | 监督微调 | Agentic RL（GRPO）|

**与 Rec-R1 等工作的区别：** QueryAgent-R1 专注于查询推荐（生成用户感兴趣的搜索词），而非物品推荐，并将商品库检索结果作为奖励信号的一部分，实现更强的业务对齐。

**创新可信度：高。** 从 CTR-CVR 对齐角度切入是工业界真实痛点，Alibaba 的大规模线上部署验证了方案可行性。

---

## 关键指标 / Key Metrics

| Dataset/Setting | Metric | QueryAgent-R1 | Baseline |
|----------------|--------|---------------|---------|
| Industrial dataset | CTR | +X% (online A/B) | Standard query rec |
| Industrial dataset | CVR | Significant improvement | Standard query rec |
| Public dataset | Query Relevance | SOTA | Prior methods |
| Production platform | DAU coverage | Tens of millions | - |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 24 | 30 | Chain-of-retrieval RL optimization is novel; consistency reward bridging CTR-CVR gap is well-motivated |
| Experimental SOTA delta | 12 | 15 | Production A/B on platform with tens of millions DAU; strong industrial validation |
| Experimental quality / ablations | 12 | 15 | Both industrial + public dataset evaluation; ablations on reward components |
| Efficiency | 6 | 10 | Memory abstraction module reduces long-context overhead |
| Generalization | 4 | 5 | Tested on proprietary + public data; broadly applicable to e-commerce search |
| Domain relevance | 24 | 25 | Direct e-commerce product recommendation and search optimization |
| **Total** | **82** | **100** | |

---

## 中文摘要

QueryAgent-R1 是 Alibaba 国际数字商业集团提出的电商查询推荐框架。现有查询推荐方法只优化查询与用户意图的相关性，导致高点击率但低转化率的问题。QueryAgent-R1 引入**链式检索优化**（Chain-of-Retrieval Optimization）：Agent 生成候选查询后，调用真实商品库进行检索，并根据检索结果验证和精化查询。训练采用 GRPO 强化学习，设计**一致性奖励**同时优化查询相关性和商品转化对齐。此外，**记忆抽象模块**从用户长期历史行为中提取兴趣图谱，高效构建用户画像。该系统已部署在日活用户数千万的大型电商平台，实现查询推荐与商品转化的端到端联合优化。
