# E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs |
| **Authors** | Xianjie Liu, Yiman Hu, Liang Wu, Ping Hu, Yixiong Zou, Jian Xu, Bo Zheng |
| **Affiliation** | Alibaba Group (Taobao / Tmall) |
| **arXiv** | [2602.08355](https://arxiv.org/abs/2602.08355) |
| **Submitted** | 2026-02-09 |
| **Venue** | arXiv preprint |
| **Domain** | E-Commerce Short Video · Multimodal Benchmark · MLLM Evaluation · Commercial Intent |
| **Bucket** | STRONG |

---

## 得分 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | First benchmark specifically for e-commerce short video understanding with commercial intent focus; multi-modal density assessment framework is novel |
| SOTA delta | 10 | 15 | Benchmark paper; reveals gaps in current MLLMs on e-commerce short video tasks |
| Exp quality / ablations | 12 | 15 | 3,961 high-quality Taobao videos; dynamic sampling for category balance; rigorous annotation pipeline |
| Efficiency | 5 | 10 | Benchmark paper; no efficiency claim |
| Generalization | 4 | 5 | Covers diverse product categories; multi-modal density assessment applicable beyond e-commerce |
| Domain relevance | 25 | 25 | Perfect fit: Taobao short videos, e-commerce content understanding, MLLM evaluation for platform use cases |
| **Total** | **78** | **100** | Essential benchmark for e-commerce short video ML; Taobao data is directly relevant |

---

## 方法概述 / Method Summary

### Story Arc
E-commerce short videos (like Taobao product ads) are fundamentally different from general-purpose short videos: they are **goal-driven** (conversion-oriented), contain **dense multimodal signals** (fast speech, overlaid text, product visuals, price callouts), and require **commercial intent reasoning** (does this video make the viewer want to buy?). Existing benchmarks focus on general video understanding tasks and fail to evaluate this unique combination of challenges.

**X is insufficient → we design Y:** General-purpose video benchmarks (Video-MME, etc.) ignore the dense multi-modal complexity and commercial reasoning requirements of e-commerce short videos → E-VAds provides the first expert-curated benchmark specifically designed for this domain, with a formal multi-modal information density assessment framework.

### Benchmark Design

```
Source: 3,961 high-quality product videos from Taobao
         ↓
[Multi-Modal Information Density Assessment]
  Quantifies: visual density (rapid scene changes, product showcases)
              audio density (speech rate, product mentions)
              textual density (overlay text frequency, call-to-action elements)
  E-commerce videos: substantially higher density than mainstream datasets
         ↓
[Dynamic Sampling Strategy]
  Improves category balance across diverse product types
  Annotation efficiency optimization
         ↓
[Annotation Tasks]
  Task 1: Commercial intent classification
  Task 2: Product attribute extraction from video
  Task 3: Open-ended commercial reasoning
  Task 4: Multi-modal signal alignment evaluation
         ↓
[MLLM Evaluation]
  26+ mainstream MLLMs evaluated on E-VAds tasks
  Reveals substantial performance gaps vs. human performance
```

### Key Product Categories Covered
- Fashion & apparel
- Consumer electronics
- Beauty & personal care
- Food & beverages
- Home goods
- …and more (wide range from Taobao)

---

## 核心指标 / Key Metrics

| Task | Best MLLM Score | Human Score | Gap |
|------|-----------------|-------------|-----|
| Commercial intent classification | Moderate | High | Significant |
| Product attribute extraction | Low-Moderate | High | Very significant |
| Open-ended commercial reasoning | Low | High | Large |
| Multi-modal density (vs. general video) | — | — | 3× higher density |

*Specific percentages not disclosed; benchmark reveals systematic underperformance of current MLLMs.*

---

## 创新分析 / Innovation Analysis

**vs. Prior Work:**
- **vs. Video-MME:** General video understanding; no e-commerce or commercial intent tasks.
- **vs. WebSRC / MMMU:** Static image/document tasks; no video temporal reasoning.
- **vs. General e-commerce benchmarks (EComStage):** EComStage focuses on text-based e-commerce tasks; E-VAds is specifically short video with full multimodal richness.
- **Unique contribution:** Multi-modal information density assessment framework — formally quantifies why e-commerce videos are harder than general videos.

**Data Quality:**
- Taobao production videos (real e-commerce, not synthetic)
- Dynamic sampling for category fairness
- Expert annotation pipeline with commercial domain knowledge

---

## 相关性评估 / Domain Willingness

直接命中：
- **短视频电商内容理解**：使用真实淘宝商品视频数据，直接对应我方业务场景
- **MLLM 评估**：揭示现有大模型在电商视频理解上的能力缺口，指导模型选型
- **达人视频质量评估**：商业意图推理任务适用于达人内容的商业价值评估
- **内容密度量化**：多模态信息密度框架可用于内容质量评分和审核优先级
- **数据标注**：注释管线设计对大规模电商视频标注有直接参考价值
