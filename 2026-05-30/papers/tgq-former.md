# TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **Authors** | (from arXiv 2605.17366, affiliations not fully disclosed in search results) |
| **Affiliation** | Industrial e-commerce research group |
| **arXiv** | [2605.17366](https://arxiv.org/abs/2605.17366) |
| **Submitted** | ~2026-05-14 |
| **Venue** | arXiv preprint |
| **Domain** | E-Commerce Recommendation · Multimodal Representation · Visual Learning |
| **Bucket** | STRONG |
| **Code** | `code/TGQ-Former/` |

---

## 得分 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 25 | 30 | Dual-stream adaptive disentanglement with metadata-anchored + exploratory query streams is novel; dual-gated vector modulation is well-motivated |
| SOTA delta | 11 | 15 | Consistent gains over strong connector baselines and end-to-end MLLMs on large-scale full-pool retrieval |
| Exp quality / ablations | 11 | 15 | Large-scale real-world e-commerce dataset, full-pool retrieval setting (harder than approximate), ablation implied |
| Efficiency | 9 | 10 | Substantially lower inference compute than end-to-end MLLMs while maintaining superior quality |
| Generalization | 3 | 5 | Specific to e-commerce I2I retrieval with noisy poster images; framework principles generalizable |
| Domain relevance | 23 | 25 | Directly addresses e-commerce product image retrieval, multimodal recommendation, product noise robustness |
| **Total** | **82** | **100** | Highly relevant; novel approach for a production-critical e-commerce problem |

---

## 方法概述 / Method Summary

### Story Arc
E-commerce product images frequently contain promotional overlays, text watermarks, and background clutter that introduce **spurious visual cues**, causing conventional visual encoders to learn irrelevant features and degrading fine-grained item-to-item (I2I) retrieval quality. Existing methods either ignore this noise or apply expensive end-to-end MLLMs that are too slow for production. TGQ-Former bridges this gap with a lightweight, text-guided dual-stream architecture.

**X is insufficient → we design Y:** Single-stream visual feature extractors cannot disentangle metadata-consistent semantic signals from noisy promotional content → TGQ-Former produces two complementary visual streams jointly guided by structured product metadata, adaptively calibrated via cross-modal agreement scoring.

### Architecture

```
Product Image
     ↓
Visual Encoder (frozen)
     ↓                        Product Metadata (text)
     ├──────────────────────────────────────────────┐
     │                                              │
     ▼                                              ▼
[Metadata-Anchored Semantic Queries]   [Text Encoder]
     ↓  captures metadata-consistent visual evidence    
[Exploratory Queries]    ← captures complementary patterns beyond metadata
     ↓
Dual-Gated Vector Modulation
   (cross-modal agreement + image-derived cues)
     ↓
Redundancy-Reduction Regularizer
   (encourages complementarity between streams)
     ↓
Fused Representation → I2I Retrieval Score
```

### Key Components
1. **Text-Guided Q-Former (TGQ-Former):** Structured product metadata serves as semantic guidance for visual token extraction, producing adaptive disentanglement into two streams.
2. **Metadata-Anchored Semantic Queries:** Capture visual evidence consistent with the metadata description (e.g., color, style, shape).
3. **Exploratory Queries:** Capture complementary visual patterns beyond what the metadata explicitly describes.
4. **Dual-Gated Vector Modulation (DGVM):** Calibrates both streams using (a) cross-modal agreement between text and visual features and (b) image-derived saliency cues.
5. **Redundancy-Reduction Regularizer (R³):** Penalizes overlap between the two streams to enforce complementarity and maximize information coverage.

---

## 核心指标 / Key Metrics

| Dataset | Metric | TGQ-Former | Baseline (Best Connector) | Baseline (End-to-End MLLM) |
|---------|--------|------------|--------------------------|---------------------------|
| Large-scale real-world e-commerce I2I | Full-pool Recall@K | **Consistent gains** | Lower | Lower but high compute |
| Inference FLOPs | Relative to E2E-MLLM | **Substantially lower** | Lower | 1× (reference) |

*Exact numbers not disclosed (proprietary dataset); qualitative claims are from paper summary.*

---

## 创新分析 / Innovation Analysis

**vs. Prior Work:**
- **vs. Single-stream Q-Former adapters:** Prior work applies a single Q-Former to extract visual features with text guidance but lacks the disentanglement mechanism for noise separation.
- **vs. End-to-end MLLMs (e.g., LLaVA-based recommenders):** Full MLLM inference is too expensive at e-commerce scale; TGQ-Former achieves comparable or better quality at a fraction of the compute.
- **vs. Masking/cropping-based noise removal:** These heuristic methods lose visual information and require manual heuristics per category; TGQ-Former learns to disentangle adaptively.

**Plausibility:** The dual-gated modulation via cross-modal agreement is a sound principle well-grounded in the contrastive learning literature. The redundancy regularizer is analogous to techniques from non-redundant feature learning. The approach is feasible for production deployment given its efficiency.

---

## 相关性评估 / Domain Relevance

直接命中我方业务场景：
- **电商图搜图（I2I检索）**：TGQ-Former 专为电商产品图的噪声鲁棒性设计
- **多模态商品理解**：利用文本元数据（标题、属性）引导视觉特征提取
- **推荐系统落地**：设计上考虑推理延迟，适合工业部署
- **内容质量**：通过分离"推广噪声"与真实产品视觉，提升特征质量

Code reproduction: `code/TGQ-Former/`
