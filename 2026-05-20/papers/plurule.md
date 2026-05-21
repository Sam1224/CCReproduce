# PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media |
| **作者** | Zoher Kachwala, Bao Tran Truong, Rasika Muralidharan, Haewoon Kwak, Jisun An, Filippo Menczer |
| **机构** | Observatory on Social Media, Indiana University; TUD Dresden University of Technology |
| **arXiv ID** | [2605.17187](https://arxiv.org/abs/2605.17187) |
| **提交日期** | May 16, 2026 |
| **代码** | Not yet public |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** Modern social media platforms are shifting toward **community-governed pluralism** — different communities (subreddits, interest groups, creator channels) define their own norms, and what violates rules in one community may be perfectly acceptable in another. Current content moderation benchmarks treat moderation as a uniform task with universal rules, ignoring this community-specific structure.

PluRule is a **multimodal, multilingual benchmark** covering **13,371 rule violations** across **1,989 Reddit communities** spanning **2,885 unique rules** in **9 languages**. Each example is a multiple-choice problem: given a comment + its surrounding thread context, identify which specific rule (if any) is violated. This mirrors how human moderators actually work. The benchmark evaluates whether models can (1) understand community-specific norms, (2) apply multi-lingual rules, and (3) handle visual context in multimodal posts.

**ZH:** PluRule 针对社交媒体的社区化规则多元性构建基准：1,989 个 Reddit 社区、2,885 条规则、9 种语言、13,371 个违规案例，每个案例为多选题形式——给定评论+上下文，识别哪条规则（若有）被违反。基准测试最新 VLM 是否具备理解社区特定规范的能力。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 现有内容审核基准（HatEval、CivilComments）使用统一的全局规则，忽视了社区驱动平台中规则高度多元化的现实（同一条评论在 r/philosophy 合规，在 r/AskScience 中可能违规）。
>
> **我们的解决方案:** PluRule 将社区特定规则理解建模为多选推理任务，规模化评估当前最优 VLM 在多语言、多社区规范下的审核能力，并揭示了巨大的性能差距。

---

## 创新性分析 / Innovation Analysis

1. **社区规范多元化建模：** 首个将平台多样化社区规则纳入评估的多模态内容审核基准。
2. **规模和覆盖度：** 1,989 社区 × 2,885 规则，远超现有单规则集基准。
3. **多语言设计：** 9 种语言，对跨语言平台治理有直接意义。
4. **vs. 先前工作：** RuleSafe-VL 关注统一平台规则推理，PluRule 关注社区异构规则理解，两者互补。

---

## 关键指标 / Key Metrics

| Model | Benchmark Acc | vs. Trivial Baseline |
|-------|---------------|----------------------|
| GPT-5.2 (best) | ~34% | +4pp |
| GPT-4o | ~31% | +1pp |
| Trivial baseline | ~30% | — |

> **Key insight:** Even GPT-5.2 barely outperforms a random baseline, revealing that pluralistic community norm understanding is an **unsolved** problem.

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 19/30 | 社区多元规范建模思路新颖，基准构建方法工程导向 |
| Experimental SOTA delta | 7/15 | 揭示差距而非提出解决方案 |
| Experimental quality | 14/15 | 规模大，多语言，多模型评估 |
| Efficiency | 3/10 | 基准论文 |
| Generalization | 5/5 | 9 语言，1989 社区 |
| Domain relevance | 22/25 | 直接对应达人/内容社区规则治理，包括中文社区 |
| **Total** | **70/100** | |
