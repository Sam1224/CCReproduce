# SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking |
| **Authors** | (Kuaishou / Production Team) |
| **Affiliation** | Kuaishou (快手) |
| **arXiv ID** | [2602.09401](https://arxiv.org/abs/2602.09401) |
| **Submission Date** | February 2026 |
| **Domain Tags** | `#recommendation` `#live-streaming` `#LLM` `#ranking` `#production` `#semantic-embeddings` |
| **Total** | **77 / 100** |

---

## 故事弧线 / Story Arc

**现有问题:** 大规模直播推荐中，内容语义建模面临两难困境：(1) **离散语义抽象**（如聚类ID）：牺牲描述精度以换取效率，内容表达粗糙；(2) **密集多模态嵌入**：独立提取，与排序优化目标弱对齐，难以捕捉细粒度内容感知信号。两种方法都无法在精度与效率之间取得良好平衡。

**设计方案:** 提出 SARM（Semantic Anchor Ranking Model），将**自然语言语义锚（Semantic Anchor）**直接嵌入排序优化流程。每个语义锚表示为**可学习文本 token**，与排序特征联合优化，使内容描述能够自适应地对齐排序目标。

---

## 方法概述 / Method Overview

### Semantic Anchor 设计

```
Live-stream Content
   ├── Video frames (visual encoder)
   ├── Audio transcript (ASR)
   └── Metadata (title, tags)
         │
         ▼
   LLM-Generated Semantic Anchor
   (Learnable text tokens, jointly optimized with ranking loss)
         │
         ▼
   ┌─────────────────────────────────────┐
   │    End-to-End Ranking Model         │
   │  User Features + Item Features +    │
   │  Semantic Anchor Tokens             │
   │         │                           │
   │    Ranking Score                    │
   └─────────────────────────────────────┘
```

### 关键创新

1. **联合优化**: Semantic Anchor tokens 与排序损失端到端反传，避免独立提取的对齐偏差
2. **内容感知的作者表示**: Anchor 以多模态内容为条件，同一主播在不同内容场景下有不同表示
3. **LLM 初始化**: 用 LLM 生成初始文本描述作为 Anchor 初始化，加速收敛
4. **实时约束感知**: 设计考虑直播实时服务的延迟约束

---

## 关键指标 / Key Metrics

| Metric | Result | Platform |
|--------|--------|---------|
| Online user engagement metrics (interaction rate) | Consistent improvement | Kuaishou (400M+ users) |
| Online retention metrics | Consistent improvement | Kuaishou |
| Negative feedback signals | Reduced | Kuaishou |
| Commercial metrics | Improved | Production A/B |

---

## 创新性分析 / Innovation Analysis

- 首次将自然语言语义锚作为可学习参数融入排序优化，打破"提取后用"的两阶段范式
- 内容感知的作者动态表示有别于传统静态 ID 嵌入
- 生产级验证（400M+ 用户 A/B 测试）赋予结论高可信度

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 21 | 30 | 端到端语义锚设计新颖；LLM 增强排序是创新方向 |
| Experimental SOTA Delta | 12 | 15 | 快手 400M+ 用户 A/B 持续提升；商业指标改善 |
| Experimental Quality | 11 | 15 | 生产级长期 A/B；多维指标验证 |
| Efficiency | 7 | 10 | 实时约束设计；可学习 token 开销可控 |
| Generalization | 4 | 5 | 结果在多个平台页面上一致 |
| Domain Relevance | 22 | 25 | 直播推荐排序；与电商直播带货高度相关 |
| **Total** | **77** | **100** | |
