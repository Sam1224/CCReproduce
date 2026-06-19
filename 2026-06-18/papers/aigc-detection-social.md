# Detecting AI-Generated Content on Social Media with Multi-modal Language Models

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Detecting AI-Generated Content on Social Media with Multi-modal Language Models |
| **Authors** | Researchers from Carnegie Mellon University and Meta |
| **Affiliation** | Carnegie Mellon University; Meta |
| **arXiv ID** | [2606.11200](https://arxiv.org/abs/2606.11200) |
| **Submitted** | June 2026 |
| **Model** | IFM-AIGCSpotter-3b (3B compact VLM) |
| **Code** | Not released (deployment on Meta platforms) |

---

## 方法概述 / Method Summary

### Story Arc

> **现有方法的问题**：现有AIGC检测方法面临三大挑战：（1）对新生成模型泛化性差；（2）依赖单一模态（纯视觉或纯文本）；（3）缺乏可解释的检测说明。这使得在真实社交媒体环境下（垃圾信息、虚假信息、身份欺诈等场景）的检测效果大打折扣。
>
> **解决方案**：提出一个持续数据策划管道，从多个社交媒体平台持续收集带社交信号（转发、举报等）的多模态图文数据对，训练一个3B紧凑视觉语言模型（VLM）用于AIGC检测与说明生成，同时利用社交上下文（文字描述、用户互动模式）强化视觉特征，实现跨平台泛化。

### Technical Approach (EN)

The paper addresses AIGC (AI-Generated Content) detection on social media platforms with three key contributions:

1. **Continuous multi-modal data curation pipeline**: Continuously collect image-text pairs with social context (captions, engagement signals, community flags) from multiple social media platforms, enabling the model to adapt to new generation methods.
2. **Compact VLM for detection + explanation**: Train IFM-AIGCSpotter-3b (3B parameters) that jointly detects AIGC and generates natural-language explanations, making moderation decisions interpretable and auditable.
3. **Social context integration**: Rather than relying on visual artifacts alone, the model leverages accompanying text, engagement patterns, and platform-specific signals — more robust against adversarial generators that remove visual artifacts.

The model is deployed on Meta social media platforms, with demonstrated downstream impact on user engagement metrics.

### 创新亮点 (ZH)

- **持续数据策划**：通过持续采集真实社交平台上带社交信号的多模态数据，使模型能自适应新的生成方法，解决泛化难题。
- **社交上下文融合**：将文字描述、用户互动模式等社交信号纳入检测，突破纯视觉方法的局限。
- **紧凑+可解释**：3B参数模型在保持SOTA检测精度的同时生成可解释说明，满足内容审核的可问责性需求。
- **生产落地**：在Meta多个平台实际部署，并验证了对用户参与度的正向影响。

---

## 关键指标 / Key Metrics

| Dataset | Metric | IFM-AIGCSpotter-3b | Best Prior |
|---------|--------|--------------------|------------|
| FakeClue (in-distribution) | Accuracy | **0.986** | ~0.96+ |
| LOKI (out-of-distribution) | Accuracy | **0.839** | < 0.80 |
| Internal social media (Meta) | Detection rate | SOTA (proprietary) | — |

---

## 评分详情 / Scoring Breakdown

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation | 20/30 | Pipeline-level innovation (continuous curation + social context) is novel at scale; core VLM training is incremental |
| Experimental SOTA delta | 13/15 | 0.986 on FakeClue, 0.839 on LOKI OOD — strong SOTA results |
| Experimental quality / ablations | 12/15 | Multiple public + internal benchmarks; cross-platform evaluation |
| Efficiency | 8/10 | 3B compact model; real-time deployment on social platforms |
| Generalization | 4/5 | Cross-platform, OOD evaluation |
| Domain relevance (ecom+governance) | 22/25 | AIGC detection is core to content governance; transferable to e-commerce live-streaming fraud |
| **Total** | **79/100** | |

---

## 差异化分析 / Novelty vs. Prior Work

| Prior Work | Gap | This Paper |
|---|---|---|
| CNNSpot, DIRE (visual-only) | Poor generalization to new generators | Continuous curation + social context |
| FakeBench, LoKI (MLLM) | No social signals, no deployment | Production VLM with social context; deployed at Meta |
| IVY-FAKE (image+video) | No text/social context | Multimodal: visual + text + social signals |
