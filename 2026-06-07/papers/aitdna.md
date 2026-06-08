## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions |
| **Authors** | Nils Dycke, Marina Sakharova, Nico Daheim, Iryna Gurevych |
| **Affiliations** | TU Darmstadt (Technical University of Darmstadt), UKP Lab |
| **ArXiv ID** | [2606.04906](https://arxiv.org/abs/2606.04906) |
| **Submitted** | 2026-06-03 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.CL |
| **Bucket** | MEDIUM |
| **Total** | **70 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | Systematic redefinition of AIGC notions + AITDNA benchmark with full genesis annotation (edit history + AI-interaction history) is a significant methodological contribution |
| Experimental SOTA delta | 10 | 15 | New benchmark; exposes gaps in existing detectors under realistic assumptions |
| Experimental quality / ablations | 11 | 15 | Detailed genesis annotation at edit history level; multiple detectors evaluated; real human-AI collaborative texts |
| Efficiency | 5 | 10 | Benchmark paper |
| Generalization | 4 | 5 | Broadly applicable to any AIGC detection scenario |
| Domain relevance | 18 | 25 | Directly applicable to platform content governance; addresses the realistic mixed-authorship scenario encountered in user-generated e-commerce content |
| **Total** | **70** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
AIGC 检测领域缺乏共识：不同研究对"什么是 AI 生成文本"的定义各不相同，benchmark 往往基于隐含假设（如"全部 AI 生成"vs."全部人类写作"），与真实场景严重脱节。真实场景中，文本往往是人类与 AI 共同创作的产物，且创作过程（edit history、AI 交互记录）对检测至关重要。

**X is insufficient → we design Y to solve it：**
> 现有 AIGC 检测 benchmark 缺乏统一的概念定义，数据集往往只保留最终文本而丢弃创作过程，使得检测器面对真实"混合创作"文本时完全没有经过验证。本文系统性地定义了多种 AIGC 的 notions（全生成、辅助写作、AI 改写等），构建 AITDNA benchmark：收集真实的人机共同创作文本，标注完整的 genesis 信息（edit history + AI interaction history），使研究者能够在明确的现实假设下评测检测器。

### 核心方法

1. **AIGC Notions 分类框架**：系统定义"AI 生成文本"的各种形式，包括：
   - 完全 AI 生成（Full AI generation）
   - AI 辅助写作（AI-assisted writing）
   - AI 改写/润色（AI rewriting/polishing）
   - 人类监督的 AI 输出（Human-supervised AI output）

2. **AITDNA Benchmark**：收集真实人机协作创作的文本，标注：
   - 完整的编辑历史（edit history）
   - 与 AI 的所有交互记录（AI-interaction history）
   - 最终文本的 AIGC notion 标签

3. **评测设计**：在 AITDNA 上评测现有检测器，揭示现有方法在"现实假设"下的失效模式。

### English Summary

The paper addresses the foundational problem of AIGC detection: the lack of a shared definition of what constitutes "AI-generated text." The authors systematically define multiple notions of AI-generated text (full generation, assisted writing, AI rewriting, etc.) and collect AITDNA, a benchmark of real human-AI co-authored texts annotated with complete genesis information including edit history and AI interaction logs. Existing detectors, evaluated on AITDNA, fail on realistic mixed-authorship texts that their training distributions do not represent. The work provides a framework for more principled benchmark construction and detector evaluation in realistic deployment scenarios.

---

## 创新点分析 / Innovation Analysis

**中文：** 核心贡献是"定义先行"——在评测之前先厘清"什么是 AIGC"，避免检测器和 benchmark 建立在隐含且不一致的假设上。AITDNA 的 genesis 标注（完整创作过程）是目前最细粒度的真实场景 AIGC 数据，对平台内容治理中的"哪些内容需要标注为 AI 辅助"具有直接参考价值。与 OpAI-Bench 的互补性强：OpAI-Bench 关注渐进式修订轨迹，AITDNA 关注真实场景下的多样创作模式。

**English:** The key contribution is definitional clarity: establishing a shared taxonomy of AIGC notions before building benchmarks or evaluating detectors. AITDNA's genesis annotation — capturing the complete creation process — is the most granular real-world AIGC dataset available. It complements OpAI-Bench: while OpAI-Bench studies controlled progressive revision trajectories, AITDNA captures the full diversity of real human-AI collaborative authorship patterns. Both are valuable for building robust content governance pipelines.

---

## 关键指标 / Key Metrics

| Setting | Detector | Performance | Notes |
|---------|----------|-------------|-------|
| Original context simulation | Multiple LLMs | Acc=77.64, Macro-F1=78.10 | On stance simulation variant |
| AITDNA overall | Existing detectors | Below baseline | Fail on realistic mixed-authorship |
| Dataset | — | 1,821 conversations | AITDNA scale |
