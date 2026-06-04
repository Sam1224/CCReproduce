# A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems |
| **Authors** | Yiming Zhu, Xu Liu, et al. |
| **Affiliation** | Meta Platforms |
| **Venue** | SIGIR 2026 (49th ACM SIGIR, Melbourne, July 20-24, 2026) |
| **arXiv** | https://arxiv.org/abs/2605.09338 |
| **Submitted** | 2026-05-10 (SIGIR 2026 proceedings paper) |
| **Domain** | Multimodal Recommendation · Large-Scale Systems · LLM · Content Understanding |
| **Code** | `code/MultimodalRecFramework/` |

---

## 方法概述 / Method Overview

### 问题 / Problem
大规模推荐系统需要处理包含图像、视频、文本的异构多媒体内容，但传统推荐系统无法充分利用多媒体内容的高维语义信号。多模态大语言模型（MM-LLM）虽具备强大的内容理解能力，但将其集成到**延迟约束的工业级架构**中面临重大挑战：推理延迟、计算成本、与现有召回/排序 pipeline 的集成。

Large-scale recommendation systems need to process heterogeneous multimedia content (image, video, text), but conventional recommenders fail to fully exploit high-dimensional semantic signals. While MM-LLMs offer powerful content understanding, integrating them into **latency-constrained industrial-scale architectures** faces major challenges: inference latency, compute cost, and integration with existing recall/ranking pipelines.

### 方法 / Method

**三角架构（Tripartite Architecture）：**

1. **内容解释层（Content Interpretation）：** 基于 LLaMA2 的多模态模型，对图像/视频内容生成描述性 caption，将多媒体内容转换为结构化文本语义表示。

2. **表示提取层（Representation Extraction）：** 从生成的 caption 和原始多媒体内容中提取融合的语义表示向量，用于下游推荐任务。

3. **系统集成层（Pipeline Integration）：** 设计与工业推荐系统的端到端集成方案，解决离线预计算 vs 在线推理的权衡，确保在延迟约束下可部署。

**Story Arc:** "传统推荐系统无法充分理解多媒体内容语义 → 三角架构将 MM-LLM 的内容理解能力集成到工业推荐 pipeline"

*Conventional recommenders cannot fully understand multimedia semantics → tripartite architecture integrates MM-LLM content understanding into industrial recommendation pipeline.*

---

## 创新性分析 / Innovation

1. **工业级落地解决方案**：不只是学术 prototype，而是针对大规模工业推荐系统（Meta 规模）的实际部署方案，提供了从 MM-LLM 到工业 pipeline 的完整集成路径。
2. **LLaMA2-based Caption 生成**：将多媒体内容转化为结构化语义 caption，解决了向量表示的多模态语义对齐问题。
3. **离线预计算设计**：通过 caption 预计算降低在线延迟，保证工业可部署性。
4. **在线 A/B 验证**：0.02% 的在线指标提升对于 Meta 量级的系统意味着数亿用户的体验改善，显示出方法的实际商业价值。

---

## 关键指标 / Key Metrics

| Dataset/System | Metric | Score | Baseline |
|----------------|--------|-------|---------|
| Meta Production System (online A/B) | Online Metric Improvement | +0.02% | Previous system |
| Offline Recommendation | Recall@K | Reported improvement | Multimodal baseline |
| Industrial Latency | Deployment Feasibility | ✓ | — |

*Note: 0.02% online improvement at Meta's scale (billions of impressions) represents significant absolute user-experience and business value.*

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 22 | Tripartite framework for industrial MM-LLM integration; novel caption-based representation extraction; production-verified approach |
| SOTA Delta | 15 | 10 | Online A/B improvement at massive scale; offline recommendation gains |
| Exp. Quality | 15 | 13 | Online A/B test + offline evaluation; industrial-scale validation at Meta |
| Efficiency | 10 | 8 | Offline precomputation design; latency-aware architecture |
| Generalization | 5 | 5 | Demonstrated at industrial scale; generalizable tripartite pattern |
| Domain Relevance | 25 | 24 | Directly addresses multimedia content understanding for large-scale recommendation — core to e-commerce content ecosystem |
| **Total** | **100** | **82** | |

---

## 与先前工作的对比 / Comparison with Prior Work

| Work | Scale | LLM Integration | Online Validated |
|------|-------|-----------------|------------------|
| VBPR, BM3 | Small/medium | No | No |
| SLMRec | Medium | Partial | No |
| HistLLM | Medium | Yes | No |
| **This Framework** | **Industrial (Meta)** | **Tripartite MM-LLM** | **Yes (A/B)** |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

本框架与电商内容生态直接相关：
- **达人内容推荐**：基于达人视频/图文的语义理解进行精准个性化推荐
- **商品内容理解**：通过 MM-LLM caption 生成商品多媒体的语义表示，支持搜索和推荐
- **内容质量评估**：caption 生成结果可用于评估内容语义质量

This framework directly applies to e-commerce content ecosystem: influencer content recommendation (semantic understanding of creator video/text), product content understanding (MM-LLM caption for semantic representation), and content quality assessment.

---

## Code Reproduction

See `code/MultimodalRecFramework/` for a faithful PyTorch reproduction of the tripartite framework.
