# LiSCP: Lightweight Stylistic Consistency Profiling for LLM-Generated Text Detection in Multimedia Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation |
| **作者** | (Authors not fully indexed in search results) |
| **机构** | Not yet confirmed |
| **arXiv ID** | [2605.05950](https://arxiv.org/abs/2605.05950) |
| **提交日期** | ~May 7, 2026 |
| **代码** | Not yet public |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** The increasing prevalence of LLMs in content creation has made distinguishing human-written text from LLM-generated text a critical task for multimedia moderation platforms. Existing detectors are brittle under adversarial paraphrasing and cross-domain distribution shift.

LiSCP constructs a **consistency profile** that combines discrete stylistic features (sentence length distribution, vocabulary diversity, punctuation patterns, POS tag ratios) with continuous semantic signals (embedding-based coherence). The key insight is that LLM-generated text maintains *stylistic consistency* across meaning-preserving perturbations, while human writing varies more naturally when paraphrased. The method generates multiple **multimodal-guided paraphrase variants** of the input text (using image/video context to anchor semantics) and measures the *stability of stylistic patterns* across these variants — high stability → LLM-generated; high variance → human-written.

**ZH:** LiSCP 构建一个**文体一致性画像**：组合离散文体特征（句长分布、词汇多样性、标点模式、词性比例）与连续语义信号（嵌入相关性）。核心洞见是 LLM 生成文本在语义保留的改写变体中保持高度文体稳定性，而人类写作则更自然地变化。方法通过多模态引导（图/视频上下文锚定语义）生成多个改写变体，测量变体间文体模式的稳定性来判断是否为 AI 生成。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 现有 LLM 文本检测器（Binoculars、Fast-DetectGPT）在跨域场景下性能大幅下降，且容易被对抗性改写绕过。
>
> **我们的解决方案:** LiSCP 通过测量文体稳定性（一个生成模型固有属性而非领域特征），构建对域漂移和对抗改写均鲁棒的轻量检测器。多模态引导改写利用了多媒体内容的视觉上下文，尤其适合新闻/电商图文混排场景。

---

## 创新性分析 / Innovation Analysis

1. **文体稳定性假设：** 将 LLM 一致性（生成时固定的统计偏好）与人类写作的自然变化对立，是有力的先验。
2. **多模态锚定：** 利用图像/视频上下文指导改写，避免语义漂移，专为多媒体内容设计。
3. **无监督特征：** 文体一致性画像不依赖大量标注数据，天然具备跨域泛化能力。
4. **vs. 先前工作：** Fast-DetectGPT 等依赖语言模型概率，容易受模型更新影响；LiSCP 基于可解释的表面文体特征，更稳定。

---

## 关键指标 / Key Metrics

| Dataset | Metric | LiSCP | Best Prior |
|---------|--------|-------|------------|
| In-domain (news/movie) | AUROC | ~94.5% | ~93.1% |
| Cross-domain transfer | AUROC | best+11.79% | baseline |
| Adversarial paraphrase | Acc | ~88% | ~71% |

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 20/30 | 文体稳定性假设新颖，多模态引导改写实用 |
| Experimental SOTA delta | 11/15 | 跨域 +11.79% AUROC 显著 |
| Experimental quality | 12/15 | 多数据集，含对抗测试 |
| Efficiency | 8/10 | 轻量化，无需大模型推理 |
| Generalization | 4/5 | 跨域强；跨语言未测 |
| Domain relevance | 20/25 | 电商图文审核中的 AIGC 文本检测 |
| **Total** | **75/100** | |
