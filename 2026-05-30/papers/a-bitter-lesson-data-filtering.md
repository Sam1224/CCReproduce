# A Bitter Lesson for Data Filtering

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | A Bitter Lesson for Data Filtering |
| **Authors** | Christopher Mohri, John Duchi, Tatsunori Hashimoto |
| **Affiliation** | Stanford University |
| **arXiv** | [2605.19407](https://arxiv.org/abs/2605.19407) |
| **Submitted** | 2026-05-19 |
| **Venue** | arXiv preprint |
| **Domain** | Data Filtering · LLM Pretraining · Data Quality · Scaling Laws |
| **Bucket** | WEAK (high-impact finding; relevant to data quality practices in our domain) |

---

## 得分 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | Counterintuitive but well-executed scaling study; "train more, filter less" in data-scarce high-compute regime challenges received wisdom |
| SOTA delta | 8 | 15 | Scaling study on smaller models; findings may not fully generalize to very large scale |
| Exp quality / ablations | 10 | 15 | Systematic scaling experiments; small dataset constraints limit scope |
| Efficiency | 7 | 10 | Implicit efficiency gain: no filtering = less preprocessing compute |
| Generalization | 4 | 5 | General pretraining finding; applicable across architectures |
| Domain relevance | 12 | 25 | WEAK: data quality is relevant to our data labeling pipeline, but paper is about LLM pretraining, not e-commerce content filtering |
| **Total** | **61** | **100** | Thought-provoking result; challenges conventional data curation wisdom |

---

## 方法概述 / Method Summary

### Story Arc
The AI training community widely believes that **filtering training data for quality is essential** — filtering out low-quality, noisy, or "distracting" data improves model performance. This belief underpins massive data curation pipelines for LLM pretraining (e.g., RefinedWeb, Dolma, FineWeb). However, the conditions under which filtering helps vs. hurts are poorly understood.

**X is insufficient → we design Y:** The conventional belief that "data filtering always helps" is based on experiments in data-rich, compute-limited regimes → new scaling studies targeting the **high-compute, data-scarce regime** show that sufficiently large models not only tolerate noisy data, but benefit from it.

### Experimental Setup

```
Scaling Study Parameters:
  Regimes tested:
    Data-rich, compute-limited: [standard setup]
    Data-scarce, compute-rich: [NEW focus — training longer on same data]
         ↓
  Filtering conditions:
    A: Heavy quality filtering (high-perplexity removal, deduplication, etc.)
    B: Light filtering (deduplication only)
    C: No filtering (raw data)
         ↓
  Models: Multiple sizes (scaling)
  Metric: Downstream task performance, perplexity
         ↓
Key Finding:
  In data-scarce regime: No filtering ≥ Heavy filtering
  Large models benefit from nominally "poor" quality data
```

### Key Finding
When compute is abundant but data is scarce (e.g., training many epochs on a fixed corpus), **the best data filter is no data filter** for large enough models. Large parameter counts provide implicit regularization that makes the model robust to noisy, low-quality, or distracting data — and that data provides additional signal when seen repeatedly.

---

## 核心指标 / Key Metrics

| Regime | Filtering | Best Model Performance |
|--------|-----------|----------------------|
| Data-rich, compute-limited | Heavy filtering | Higher |
| Data-scarce, compute-rich | No filtering | Higher or Equal |
| Data-scarce, compute-rich | Heavy filtering | Lower |

---

## 创新分析 / Innovation Analysis

**vs. Prior Work:**
- **vs. DCLM, FineWeb, RefinedWeb:** These assume filtering always helps; this paper shows the regime matters critically.
- **vs. Bitter Lesson (Sutton 2019):** Sutton argued that scale trumps hand-crafted features; this paper extends to data curation: scale also trumps hand-crafted filtering.
- **Practical implication:** For platforms building domain-specific LLMs on proprietary data (which is inherently limited in scale), spending heavily on data filtering may be counterproductive.

**Relevance to Our Domain:**
- If building e-commerce-specific LLMs from proprietary data (product listings, reviews, live-stream transcripts), aggressive filtering may not be warranted.
- Focus compute budget on model size and training steps rather than elaborate data curation pipelines.
- Exception: de-duplication likely still helps; the finding is strongest for quality-based filtering.

---

## 相关性评估 / Domain Relevance

间接相关：
- **数据质量策略**：对我方使用私域电商数据训练垂直领域模型具有方法论指导意义
- **大规模数据标注**：质量过滤是否必要的问题在我方标注管线中同样存在
- **LLM 预训练**：若构建电商垂直 LLM，此结论影响数据管线决策
- **实践影响小**：我方更关注数据标注质量（监督学习），而非无监督预训练，直接影响有限
