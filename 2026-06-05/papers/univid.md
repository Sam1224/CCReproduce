# UNIVID: Unified Vision-Language Model for Video Moderation

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| Title | UNIVID: Unified Vision-Language Model for Video Moderation |
| arXiv | [2606.05748](https://arxiv.org/abs/2606.05748) |
| Submitted | ~June 4–5, 2026 |
| Authors | Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao |
| Affiliation | **ByteDance** |
| Venue | arXiv preprint |
| Code | `code/UNIVID/` |
| Domain Tag | content-governance · video-moderation · VLM · e-commerce |

---

## 方法概述 / Method Summary

**English:**  
UNIVID addresses a dual challenge in platform-scale video content moderation: production systems rely on hundreds of fragmented, policy-specific classifiers that are expensive to maintain, opaque, and brittle to policy changes. UNIVID instead trains a single unified vision-language model that generates **policy-aware captions** as an interpretable intermediate representation. These captions encode which policy rules a video potentially violates; downstream binary decisions are derived from the captions, making audit trails transparent and multi-policy reuse trivial. Training mixes expert human-refined annotations with LLM-generated synthetic labels aligned to safety guidelines, explicitly avoiding safety-guardrail refusals that plague vanilla VLMs.

**中文：**  
现有平台级视频内容审核系统依赖数百个独立的特定策略分类器，难以维护、不透明、对策略变更脆弱。UNIVID 将整个审核流程统一为一个视觉-语言模型，通过生成**政策感知字幕（policy-aware caption）**作为可解释中间表示。字幕明确描述视频可能违反哪些政策条款，下游的违规判定直接从字幕推断，使审核链路透明可追溯，且一模型可复用于多政策场景。训练数据混合人工精标和合成标签，特别规避了通用 VLM 的安全护栏拒识问题。

---

## 故事弧线 / Story Arc

> **传统方案的不足 →** 现有视频审核依赖 1000+ 个策略专用分类器，维护开销极高，模型决策不透明，且无法复用于新政策。  
> **我们的方案 →** 设计 UNIVID：单一 VLM，以政策感知字幕为核心输出，用于全政策统一审核与可解释决策。

---

## 创新点 / Innovation

1. **字幕作为中间表示（Caption-as-Proxy）：** 将视频内容违规审核转化为条件字幕生成，字幕本身即审核理由；与传统多分类器相比，可解释性大幅提升。
2. **合成+人工混合训练数据配方：** 利用合成数据扩展稀缺违规类别的覆盖，同时用人工精标保证边界案例质量；明确解决 VLM 安全护栏误拒问题。
3. **替代 1000+ 模型的统一骨干：** 单一模型覆盖所有政策，相比多分类器范式显著降低计算和工程成本。
4. **可解释+可追溯审核链路：** 字幕输出可直接面向人工审核员，支持一键查证、申诉辅助。

**差异化 vs 先前工作：**  
VideoModerator（SIGIR 2021）等先前工作仍采用多头分类范式；UNIVID 将生成式 VLM 用于内容审核，并以字幕替代 logit，是方法论层面的范式转变。

---

## 关键指标 / Key Metrics

| Dataset / Setting | Metric | UNIVID | Baseline |
|-------------------|--------|--------|----------|
| ByteDance production platform | Violation Leakage ↓ | **−42.7% (relative)** | Existing system |
| ByteDance production platform | Overkill Rate ↓ | **−37.0% (relative)** | Existing system |
| Engineering cost | Models replaced | 1 (UNIVID) vs **1000+** classifiers | — |

---

## 评分 / Scoring

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation (max 30) | 25 | Policy-aware caption as moderation proxy is genuinely novel; unification at production scale is rare |
| SOTA Delta (max 15) | 13 | −42.7% leakage / −37% overkill are large relative improvements on real traffic |
| Experimental Quality (max 15) | 11 | Production deployment is strong evidence, but ablation table details not publicly available |
| Efficiency (max 10) | 10 | 1 model replaces 1000+ — massive engineering saving |
| Generalization (max 5) | 4 | Multi-policy coverage; video + multi-modal inputs |
| Domain Relevance (max 25) | 24 | Core platform content governance, directly applicable to e-commerce live & short video |
| **Total** | **87** | **Score ≥ 80 → code reproduction required** |

---

## 代码复现 / Code Reproduction

→ `code/UNIVID/`

The implementation follows the UNIVID architecture: a vision-language backbone fine-tuned with a policy-caption objective. The toy pipeline demonstrates:

1. **Multi-frame visual encoding** from video clips
2. **Policy-conditioned caption generation** (each policy rule injected as a prefix prompt)
3. **Violation detection** derived from caption content via a lightweight classifier head
4. **Mixed-data training** with synthetic + human labels

See `code/UNIVID/README.md` for quickstart.
