## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Operation-Guided Progressive Human-to-AI Text Transformation Benchmark for Multi-Granularity AI-Text Detection |
| **Authors** | Sondos Mahmoud Bsharat, Jiacheng Liu, Xiaohan Zhao, Tianjun Yao, Xinyi Shang, Yi Tang, Jiacheng Cui, Ahmed Elhagry, Salwa K. Al Khatib, Hao Li, Salman Khan, Zhiqiang Shen |
| **Affiliations** | Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), University College London |
| **ArXiv ID** | [2606.06481](https://arxiv.org/abs/2606.06481) |
| **Submitted** | 2026-06-04 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.CL, cs.AI |
| **Code** | [VILA-Lab/OpAI-Bench](https://github.com/VILA-Lab/OpAI-Bench) (official) |
| **Bucket** | STRONG |
| **Total** | **82 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 25 | 30 | Progressive trajectory (v0–v8) as benchmark unit + 5 edit operations + multi-granularity provenance is genuinely novel framing; non-monotonic detectability finding is a significant empirical contribution |
| Experimental SOTA delta | 12 | 15 | Reveals blind spots of 17 detectors (8 doc + 7 sentence + 2 fine-grained); exposes non-monotonic failure patterns missed by all existing benchmarks |
| Experimental quality / ablations | 13 | 15 | 4 domains, 9 revision versions per sample, controlled AI coverage, comprehensive cross-granularity evaluation |
| Efficiency | 5 | 10 | Benchmark paper; evaluation cost scales with number of detectors |
| Generalization | 5 | 5 | Applicable to any domain and AIGC detection model; cross-domain by construction |
| Domain relevance | 22 | 25 | AIGC detection is a core requirement for platform content governance; directly applicable to detecting AI-generated product descriptions, reviews, and marketing copy |
| **Total** | **82** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
AI 写作助手的普及使"纯人类写作"与"纯 AI 生成"之间的界限日益模糊：大量真实文档是人类与 AI 协同编辑的产物，且修订过程是渐进式的（每次只改一部分）。现有 AIGC 检测 benchmark 仅关注"最终版本"，完全忽略了"修订轨迹"和"操作类型"对可检测性的影响。

**X is insufficient → we design Y to solve it：**
> 现有 benchmark 用独立采样的纯人/纯 AI 文档对，无法捕捉"中等 AI 覆盖率"下检测器的行为，也无法区分同等覆盖率下不同编辑操作（润色 vs. 压缩 vs. 改写）导致的不同检测难度。OpAI-Bench 将"修订轨迹"作为基本单元，为每篇文档生成 9 个版本（v0=全人类, v1–v8=逐步增加 AI 覆盖），并附带 5 类操作标记和文档/句子/Token/Span 四粒度溯源，揭示了"非单调可检测性"这一关键新现象。

### 核心方法

1. **渐进轨迹构建**：从人类原稿（v0）出发，在 5 类 AI 编辑操作（polish/paraphrase/style-rewrite/compress/expand）下按固定 AI 覆盖比例（0%→100%）递增生成 v1–v8，每个版本记录哪些 token/span 被 AI 修改。

2. **多粒度溯源标注**：每个版本同时携带：
   - 文档级 AI 覆盖率标签
   - 句子级 AI/Human 标签
   - Token 级和 Span 级边界标注

3. **Benchmark 评测**：在 4 个领域（essays/reports/news/abstracts）上系统评测 17 个检测器（RADAR、Fast-DetectGPT、Gemini-Flash 等），分析跨版本、跨操作、跨领域的检测性能变化。

4. **关键发现**：可检测性不随 AI 覆盖单调上升；"中等覆盖+压缩操作"往往是最难检测的区间；不同操作类型和领域对可检测性的影响远超 AI 覆盖比例本身。

### English Summary

OpAI-Bench introduces a progressive human-to-AI text transformation benchmark that treats the revision trajectory — not just the final output — as the core unit of analysis. Starting from a human-written document (v0), it constructs nine progressively AI-edited versions (v1–v8) under five edit operations (polish, paraphrase, style-rewrite, compress, expand) at controlled AI coverage levels, while preserving multi-granularity provenance at document, sentence, token, and span levels. The benchmark is evaluated with 17 detectors across 4 domains, revealing the critical finding that detectability is non-monotonic with respect to AI coverage — intermediate-coverage compression stages are often harder to detect than both purely human and heavily AI-edited endpoints.

---

## 创新点分析 / Innovation Analysis

**中文：** 核心创新是把"修订轨迹"（cumulative version chain）作为 benchmark 一等公民，而非独立采样的文档对。这使得研究者首次能够：(a) 控制 AI 覆盖率研究其对可检测性的因果影响；(b) 分离"操作类型"与"覆盖率"两个变量；(c) 在多粒度上对同一文档的同一修订进行对比分析。"非单调可检测性"发现是对现有检测器能力边界的重要实证揭示，对平台内容治理中的 AIGC 检测策略设计有直接指导意义。

**English:** The key novelty is the trajectory-as-unit design: by constructing cumulative revision chains rather than sampling paired documents, OpAI-Bench disentangles AI coverage from edit operation type, enabling the first controlled analysis of how both factors independently shape detectability. The non-monotonic finding — where intermediate-coverage compression defeats detectors that handle fully AI-written text — is a directly actionable insight for deploying AIGC detectors in production content governance pipelines.

---

## 关键指标 / Key Metrics

| Setting | Detector | Metric | v1 | v8 |
|---------|----------|--------|----|----|
| essays | RADAR | F1-AI | 50.4 | 71.5 |
| reports | Fast-DetectGPT | F1-AI | 61.3 | 29.1 |
| news | Fast-DetectGPT | F1-AI | 65.0 | 35.2 |
| essays | Gemini-Flash (LLM-det) | F1-AI | 9.5 (v1) | 57.9 (v8); peak v3=71.5 |

*Non-monotonic pattern: intermediate versions (v3–v5) often harder than v1 or v8 for certain detectors/domains.*

---

## 代码复现 / Code Reproduction

官方代码已由作者发布于 GitHub: [VILA-Lab/OpAI-Bench](https://github.com/VILA-Lab/OpAI-Bench)。该仓库包含完整的 benchmark 构建脚本、数据格式、检测器评测接口。无需本仓库另行复现。
