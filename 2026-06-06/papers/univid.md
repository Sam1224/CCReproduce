# UNIVID: Unified Vision-Language Model for Video Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | UNIVID: Unified Vision-Language Model for Video Moderation |
| **Authors** | Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao |
| **Affiliation** | ByteDance |
| **arXiv** | [2606.05748](https://arxiv.org/abs/2606.05748) |
| **Submitted** | June 4, 2026 (indexed June 5–6, 2026 GMT+8) |
| **Code** | `code/UNIVID/` (reproduced; official repo not released at time of report) |
| **Bucket** | STRONG |

---

## 故事弧线 / Story Arc

> **"传统视频审核依赖数千个黑盒分类器，碎片化、不可维护、缺乏透明度 → 我们设计 UNIVID，用单一视觉语言模型生成以政策为导向的字幕作为可解释中间表示，支撑全链路视频审核系统。"**

Traditional content moderation platforms rely on thousands of fragmented, policy-specific binary classifiers. Each category (violence, adult content, spam, etc.) requires a separate model, leading to maintenance overhead, inconsistent reasoning, and no human-verifiable decision trace. UNIVID replaces this fragmented stack with a single vision-language captioner that generates structured, policy-aware captions. These captions act as a shared, interpretable representation for all downstream moderation tasks and can be cached for embedding-based similarity search.

---

## 方法 / Method

UNIVID introduces a **three-stage cascaded moderation pipeline**:

```
Video Input
    │
    ▼
(A) Risk Filter
    ├── Multi-modal risk funnel
    ├── Fuses UNIVID caption with visual/audio signals
    └── Filters potential high-risk videos

    │
    ▼
(B) Moderation Actor
    ├── UNIVID-Lite  → lightweight fine-tuned model for high-throughput moderation decisions
    └── UNIVID-RAG  → retrieval-augmented model that recalls similar prior violative events

    │
    ▼
(C) Trend Governance
    ├── Caches UNIVID embedding vectors
    ├── Clustering over embeddings to detect emerging violation trends
    └── Adaptive trend head for novel risk types
```

**Training Data Recipe**:
- Expert human-refined labels for known violation categories
- Synthetic data for low-resource policy areas
- Alignment procedure to prevent safety-guardrail refusals while maintaining nuanced policy adherence

**Key insight**: Captions serve dual purpose — human-verifiable text for audit trail **and** dense vector representation for downstream tasks (classification, similarity, clustering).

---

## 创新性 / Innovation

| Aspect | Prior Work | UNIVID |
|--------|------------|--------|
| Architecture | 1000+ separate binary classifiers | Single unified VLM backbone |
| Interpretability | Black-box scores | Structured policy-aware captions |
| Leakage handling | Static rules / periodic retraining | UNIVID-RAG recall + Trend Governance |
| Emerging violations | Requires new model training | Embedding clustering + adaptive trend head |
| Maintenance | N models × N policies | One backbone, policy-aware prompting |

**Novelty assessment**: High plausibility. The captioner-as-moderator paradigm is well-motivated by the success of LLMs in zero/few-shot classification through generation. The three-stage pipeline (filter → actor → trend) mirrors real platform needs. The use of cached UNIVID embeddings for trend detection is a practical industrial contribution.

---

## 关键指标 / Key Metrics

| Dataset/Context | Metric | UNIVID | Baseline |
|----------------|--------|--------|---------|
| Industrial production deployment | Violation Leakage Rate | **↓42.7% (relative)** | — |
| Industrial production deployment | Overkill Rate | **↓37.0% (relative)** | — |
| Operational | Number of policy-specific models | **~0** (single backbone) | 1,000+ |

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 25 | Unified captioner paradigm replacing 1000+ models is highly novel for industrial moderation; trend governance via embeddings adds depth |
| Experimental SOTA delta | 15 | 13 | 42.7% leakage reduction is substantial production impact, though absolute numbers not disclosed |
| Experimental quality / ablations | 15 | 11 | Production A/B test evidence is strong; paper reports cascaded ablation across the three stages |
| Efficiency | 10 | 9 | Retiring 1000+ models is a massive engineering efficiency win; cached embeddings avoid re-inference |
| Generalization | 5 | 4 | Tested across multiple violation categories on live platform; not yet demonstrated on third-party benchmarks |
| Domain relevance (ecom + governance) | 25 | 25 | ByteDance, video platform content moderation, influencer governance — perfect alignment |
| **Total** | **100** | **87** | Industrial-scale content governance system with strong novelty and production validation |

---

## 复现说明 / Reproduction Notes

Full PyTorch implementation at `code/UNIVID/`. Covers:
- `model/univid.py` — VLM backbone with LoRA + policy-aware caption head
- `data/dataset.py` — toy moderation dataset loader
- `train.py` — two-stage training (SFT on captions → policy fine-tune)
- `pipeline/risk_filter.py` — Risk Filter stage
- `pipeline/moderation_actor.py` — UNIVID-Lite + UNIVID-RAG
- `pipeline/trend_governance.py` — embedding clustering for trend detection
- `eval.py` — leakage / overkill evaluation
