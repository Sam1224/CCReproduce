# RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation |
| **作者** | Zhifeng Lu, Dianyun Wang, Yuhu Shang, Zhenbo Xu |
| **机构** | Beijing University of Posts and Telecommunications, Beijing, China |
| **arXiv ID** | [2605.07760](https://arxiv.org/abs/2605.07760) |
| **提交日期** | May 8, 2026 |
| **代码** | N/A (benchmark paper; reproduction at `code/RuleSafe-VL/`) |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** Platform content moderation applies explicit policy rules and context-dependent conditions to decide whether user content is allowed, restricted, or removed. Existing multimodal safety benchmarks reduce moderation to predicting predefined labels, completely ignoring the underlying rule structure. A high score on such benchmarks tells us nothing about whether a model applies the policy correctly or shortcuts to the right label via superficial cues.

RuleSafe-VL introduces a benchmark for **rule-conditioned decision reasoning** in vision-language content moderation. Derived from publicly available platform moderation policies, it formalizes **93 atomic rules** and **92 typed rule relations** (e.g., conditional overrides, exceptions, scope constraints), yielding **2,166 expert-reviewed image-text cases** across three high-risk policy families: (1) nudity / sexualized content, (2) dangerous / harmful behavior, and (3) graphic / injury-related content. Each case provides the image, the accompanying text context, and the set of applicable rules; the model must identify which rule(s) are violated and provide a structured reasoning chain. The evaluation separates rule-retrieval accuracy, rule-application accuracy, and final moderation outcome accuracy.

**ZH:** 平台内容审核依赖显式策略规则对内容进行允许/限制/删除判定。现有多模态安全基准把审核退化为标签预测，完全忽略了底层的规则结构。RuleSafe-VL 提出一种**规则条件化决策推理**基准——从公开平台审核政策中形式化提炼出 93 条原子规则和 92 种规则关系，构建 2,166 个专家审核的图文案例，涵盖三大高危政策族（色情/性化内容、危险/有害行为、暴力/伤害内容）。每个案例包含图片、文本上下文和适用规则集，模型需识别违规规则并提供结构化推理链。评估分别测量规则检索、规则应用与最终审核决策三个层面的准确率。

---

## 故事主线 / Story Arc

> **现有方法的不足 (X is insufficient):** 当前多模态安全基准（如 VLGuard、SPA-VL）仅测试模型是否输出正确的安全/不安全标签，而不关心模型是否真正理解并应用了规则。即使是以"快捷方式"通过表面特征做出判断的模型也能取得高分。
>
> **我们的解决方案 (we design Y):** RuleSafe-VL 将内容审核建模为**规则条件化推理**任务：给定案例和规则集，模型必须追踪哪些规则被激活、规则如何交互、证据是否充分，才能做出正确的审核决策。这要求模型具备真正的政策理解能力，而非标签匹配能力。

---

## 创新性分析 / Innovation Analysis

1. **规则结构建模：** 首个将审核政策形式化为规则图（93 个原子规则 + 92 种类型化关系）的视觉-语言基准。
2. **多粒度评估：** 分离规则检索、规则应用、最终决策三层评估，能定位模型失败的具体环节。
3. **专家审核案例：** 2,166 个案例均经人工审核，避免了自动生成数据集的标注噪声。
4. **vs. 先前工作：** VLGuard、SPA-VL、MLLMGuard 等均为二分类任务，RuleSafe-VL 是首个要求规则推理链的基准。

**Plausibility:** 高。规则推理是真实平台审核员的工作方式，将其引入评估完全合理。

---

## 关键指标 / Key Metrics

| Dataset | Metric | Best Model (GPT-5.2) | vs. Trivial Baseline |
|---------|--------|-----------------------|----------------------|
| RuleSafe-VL (2166 cases) | Rule-Retrieval Acc | ~58% | +28pp |
| RuleSafe-VL | Rule-Application Acc | ~51% | +21pp |
| RuleSafe-VL | Final Moderation Acc | ~64% | +14pp |
| RuleSafe-VL | Full-Chain Acc | ~38% | — |

> *All values approximate from paper's reported results; exact numbers may differ.*

**Key finding:** Even GPT-5.2 with chain-of-thought fails to correctly apply the full rule chain in ~62% of cases, revealing a fundamental gap in current VLM rule reasoning.

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 24/30 | 首个规则条件化内容审核基准，规则图形式化新颖 |
| Experimental SOTA delta | 9/15 | 诊断性基准，揭示差距而非提出更优模型 |
| Experimental quality | 14/15 | 2166 专家案例，三大风险族，多模型对比 |
| Efficiency | 4/10 | 基准论文，未涉及推理效率 |
| Generalization | 5/5 | 三类内容类型，9+种规则交互，覆盖全面 |
| Domain relevance | 25/25 | 直接对应电商平台内容治理与违规检测核心需求 |
| **Total** | **81/100** | |

**Code reproduction:** `code/RuleSafe-VL/` — 包含基准数据格式、规则图解析器、模型评估流程及指标计算的完整 PyTorch 实现。
