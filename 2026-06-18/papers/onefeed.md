# OneFeed: A Unified Generative Framework for Feed Content Enhancement and Query Generation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | OneFeed: A Unified Generative Framework for Feed Content Enhancement and Query Generation |
| **arXiv ID** | [2606.07972](https://arxiv.org/abs/2606.07972) |
| **Submitted** | June 6, 2026 |
| **Venue** | Preprint |
| **Code** | Not publicly released |

---

## 方法概述 / Method Summary

### Story Arc

> **现有方法的问题**：Feed推荐和搜索系统长期分离运作——Feed主要捕捉用户的隐式兴趣（浏览行为），搜索依赖用户显式输入的查询来检索意图匹配的内容。这种分离导致用户理解碎片化，错失了利用Feed交互改进查询生成、用生成查询增强Feed候选检索的机会。
>
> **解决方案**：提出OneFeed，一个统一生成框架，通过共享行为编码器编码异构用户行为序列，并采用两个生成头：Feed语义ID生成器（生成内容语义ID用于推荐检索）和意图查询生成器（生成自然语言查询用于搜索候选扩展），同时引入SID-Query对齐目标在内容语义ID和查询表征之间学习共享语义空间。

### Technical Approach (EN)

OneFeed unifies feed content recommendation and search query generation within a single generative model:

1. **Shared Behavior Encoder**: Encodes heterogeneous user behavior sequences (views, clicks, searches, purchases) into a shared representation, enabling cross-task knowledge transfer.
2. **Feed Semantic ID Generator**: Generates content Semantic IDs for recommendation retrieval — enables generative recommendation without item ID lookup tables.
3. **Intent Query Generator**: Generates natural-language queries from user behavior context for search-based candidate expansion, bridging the feed→search gap.
4. **SID-Query Alignment**: An alignment objective that forces the model to learn a shared semantic space between content IDs (recommendation domain) and query representations (search domain), closing the semantic gap between the two systems.

### 创新亮点 (ZH)

- **统一推荐+搜索**：单一模型同时处理Feed推荐和查询生成，打通两个长期分离的系统。
- **SID-Query对齐**：设计专用对齐目标连接内容语义ID与自然语言查询的表征空间，是核心技术创新。
- **异构行为建模**：统一编码浏览、搜索、购买等多种异构行为，丰富用户理解。

---

## 关键指标 / Key Metrics

| Metric | Setting | Result |
|--------|---------|--------|
| Feed recommendation recall | Offline | Improved vs. separate feed model |
| Query generation quality | Downstream product CVR | Improved vs. separate search model |
| SID-Query alignment | Semantic similarity | Significant improvement |

---

## 评分详情 / Scoring Breakdown

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation | 22/30 | Novel unification of feed+search with SID-Query alignment; elegant cross-system generative approach |
| Experimental SOTA delta | 11/15 | Outperforms separate systems; numbers proprietary |
| Experimental quality / ablations | 11/15 | Ablations on alignment objective and generator heads |
| Efficiency | 7/10 | Single model replaces two; deployment efficiency gain |
| Generalization | 4/5 | Applicable to any platform with both feed and search |
| Domain relevance (ecom+governance) | 22/25 | Feed recommendation, content enhancement, query generation — core to e-commerce content ecosystem |
| **Total** | **77/100** | |
