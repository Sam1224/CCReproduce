# SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking |
| **Authors** | Ruochen Yang, Yueyang Liu, Zijie Zhuang, Changxin Lao, Yuhui Zhang, Jiangxia Cao, Jia Xu, Xiang Chen, Haoke Xiao, Xiangyu Wu, Xiaoyou Zhou, Xiao Lv, Shuang Yang, Tingwen Liu, Zhaojie Liu, Han Li, Kun Gai |
| **Affiliation** | Kuaishou Technology; Institute of Information Engineering, Chinese Academy of Sciences |
| **arXiv** | [2602.09401](https://arxiv.org/abs/2602.09401) |
| **Submitted** | 2026-02-10 |
| **Venue** | arXiv preprint (Kuaishou Technology, production deployment) |
| **Domain** | Live-Streaming Recommendation · LLM · Ranking · Semantic Representation |
| **Bucket** | STRONG |
| **Code** | `code/SARM/` |

---

## 得分 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 25 | 30 | Semantic anchor as learnable text tokens jointly optimized with ranking — novel integration of LLM semantics into ranking objectives; adapts content description to ranking signal rather than just using pre-computed embeddings |
| SOTA delta | 11 | 15 | Deployed at Kuaishou (billions of users); measurable gains in online A/B experiments on live-streaming recommendation |
| Exp quality / ablations | 12 | 15 | Production-scale offline + online experiments; ablations on anchor granularity and text token design |
| Efficiency | 8 | 10 | End-to-end ranking with semantic anchors; LLM used offline for anchor generation; online serving maintains real-time constraints |
| Generalization | 3 | 5 | Live-streaming specific; principles transfer to short-video and e-commerce live ranking |
| Domain relevance | 24 | 25 | Direct hit: live-streaming ranking at Kuaishou scale, 达人 content semantics, real-time recommendation |
| **Total** | **83** | **100** | Industry paper with production validation; highly relevant to live-stream e-commerce ranking |

---

## 方法概述 / Method Summary

### Story Arc
Live-streaming recommendation must model **non-stationary content semantics** in real time — the content of a live stream evolves every second (host speaking, product demonstrations, audience interactions), yet conventional ranking models rely on static ID embeddings and shallow behavioral signals that cannot capture this linguistic and visual dynamism. Furthermore, LLM-generated descriptions are expensive to serve online at ranking latency.

**X is insufficient → we design Y:** Static ID-based ranking models miss the rich conversational/visual semantics of live streams → SARM introduces learnable **Semantic Anchors** represented as text tokens jointly optimized with ranking objectives, enabling fine-grained author representations that adapt to live content without online LLM inference.

### Architecture

```
Live-Stream Content (multimodal: speech, video frames, interaction text)
         ↓
[Offline LLM Processing]
  Generates textual semantic anchors per anchor granularity
  Each anchor = a learnable set of text tokens
         ↓
[Semantic Anchor Representation]
  Text tokens optimized jointly with ranking features
  Captures: product mentions, host style, topical focus
         ↓
[End-to-End Ranking Model]
  Integrates semantic anchors with traditional ranking features
  (behavioral signals, user profile, context)
         ↓
Real-time Live-Streaming Ranking Score
```

### Key Technical Contributions
1. **Semantic Anchors:** Learnable text tokens that represent a compact, ranking-objective-aligned description of live-stream content. Unlike pre-computed frozen embeddings, anchors are jointly trained with the ranking objective.
2. **LLM-Generated Anchor Initialization:** An offline LLM generates initial anchor content from live-stream transcripts and metadata; anchors then evolve during training.
3. **Anchor Granularity Control:** Authors systematically ablate the granularity of anchors (per-session, per-topic, per-product) to find the optimal representation level.
4. **Non-stationary Content Adaptation:** Anchors refresh as live content evolves, enabling the ranking model to track content drift within a session.
5. **End-to-End Training:** The entire pipeline — from anchor representation to ranking score — is jointly optimized, avoiding the two-stage mismatch of separate embedding + ranking systems.

---

## 核心指标 / Key Metrics

| Metric | Setting | Improvement |
|--------|---------|-------------|
| Live-streaming recommendation quality | Kuaishou A/B test (online) | Significant improvement in watch time and click-through |
| Content semantic alignment | Offline evaluation | Substantial improvement over baseline ranking models |
| Semantic richness | Anchor ablation | Per-topic granularity outperforms session-level and product-level anchors |

*Specific numerical deltas are proprietary; gains confirmed in production deployment at Kuaishou.*

---

## 创新分析 / Innovation Analysis

**vs. Prior Work:**
- **vs. Static content embeddings (CLIP/BERT pre-computed):** Pre-computed embeddings are fixed; semantic anchors are jointly optimized with the ranking objective, preventing feature-task mismatch.
- **vs. Real-time LLM inference for ranking:** Eliminates online LLM cost by moving semantic processing offline while maintaining semantic richness via learnable tokens.
- **vs. DIN / DIEN (sequential modeling):** Traditional sequence models capture user behavior history but not live-content semantic dynamics; SARM adds the content semantic axis.
- **vs. NOVA / content-aware ranking:** More explicit semantic grounding via LLM-generated anchors + end-to-end optimization vs. contrastive content pre-training.

**Plausibility:** The idea of representing live-stream semantics as learnable text tokens is elegant and grounded in the parameter-efficient fine-tuning literature (similar to prompt tuning). Production deployment at Kuaishou scale validates feasibility.

---

## 相关性评估 / Domain Relevance

核心命中我方达人生态/直播场景：
- **直播间排序**：直接命中直播推荐场景，解决语义实时性与延迟约束的矛盾
- **达人内容语义理解**：语义锚点可捕捉达人风格、选品偏好、讲解方式等关键特征
- **商品内容标注**：LLM 生成锚点的思路可迁移到商品/达人内容自动标注
- **内容治理**：语义锚点对异常内容（违规话语、诱导消费）检测有潜在应用

Code reproduction: `code/SARM/`
