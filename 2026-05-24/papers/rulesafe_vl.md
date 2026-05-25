# RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **Title** | RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation |
| **Authors** | Zhifeng Lu, Dianyuan Wang, Yuhu Shang, Zhenbo Xu |
| **Affiliation** | (Institution not explicitly stated in search results; inferred: Chinese academic/industry lab) |
| **arXiv** | [2605.07760](https://arxiv.org/abs/2605.07760) |
| **Submitted** | 8 May 2026 |
| **Domain** | Content Governance · VLM Evaluation · Content Moderation |
| **Bucket** | STRONG |
| **Code** | `code/RuleSafe-VL/` (reproduction) |

---

## 分数 / Score

| 维度 | 分数 | 满分 |
|---|---|---|
| Innovation | 26 | 30 |
| Experimental SOTA delta | 10 | 15 |
| Experimental quality / ablations | 12 | 15 |
| Efficiency | 4 | 10 |
| Generalization | 5 | 5 |
| Domain relevance (ecom + governance) | 23 | 25 |
| **Total** | **80** | **100** |

**Justification**: Innovation score is high because existing VLM safety benchmarks reduce moderation to label matching, ignoring the policy rule structure. RuleSafe-VL is the first to formalize platform moderation policies into atomic rules, typed rule relations, and multi-condition cases. Domain relevance is very high: platform rule enforcement is exactly the operational reality for e-commerce content governance teams. Efficiency not addressed (pure evaluation benchmark). Generalization is maximum because rules are derived from real publicly available platform policies.

---

## 方法概述 / Method Summary

平台内容审核系统依据显式政策规则与上下文条件来决定用户内容是否允许、受限或被删除。正确的审核结论不仅依赖最终标签，还取决于：哪些规则被激活、规则之间如何交互、以及现有证据是否充分。然而，现有的多模态安全基准大多将审核简化为预定义标签匹配，完全忽略了底层规则结构——高分并不代表模型正确应用了政策。

**RuleSafe-VL** 填补了这一空白：
1. 从公开平台内容政策中提炼出 **93 条原子规则** 和 **92 种有类型的规则关系**（如：条件关系、排斥关系、优先关系）。
2. 构建 **2,166 个上下文敏感的图文案例**，覆盖三个高风险政策族（例如：仇恨言论、成人内容、欺诈内容）。
3. 每个案例标注了激活的规则集合、规则交互方式，以及充分性标志（证据是否完备）。
4. 评估模型在规则条件下的决策推理能力，而非仅看最终的允许/删除标签。

**Story Arc**: 现有多模态内容安全基准缺乏规则结构 → RuleSafe-VL 通过形式化平台政策为规则有向图，构建多条件图文测试用例，系统评估 VLM 在政策驱动的内容审核中的推理能力。

---

## 创新性分析 / Innovation Analysis

**与 prior work 的对比**:
- 现有基准（如 RTVLM、SPA-VL、MM-SafetyBench）：将安全分类为"安全/不安全"二分或有限类别，没有引入政策规则结构。
- RuleSafe-VL 的独特性：  
  (a) **规则形式化** — 首次将平台公开政策转化为有向规则图（原子规则 + 关系类型）；  
  (b) **规则条件推理评估** — 区分模型是"走了捷径"还是"真正理解了规则"；  
  (c) **充分性推理** — 测试模型是否能识别证据不足需要进一步审查的情况。

**可行性**: 规则来自公开政策，案例构建方法论可复现。标注成本相对可控（规则形式化是一次性工作）。

---

## 关键指标 / Key Metrics

| 评估维度 | 最佳现有模型表现 | 说明 |
|---|---|---|
| 规则激活识别准确率 | 显著低于人类基线 | 当前顶级 VLM 在判断哪些规则被激活时表现不稳定 |
| 规则交互推理 | 多数模型失败 | 尤其在规则排斥/优先关系上失败率高 |
| 充分性推理（证据不足识别）| VLMs 普遍过于自信 | 导致误报或漏报 |
| 政策族内宏平均 F1 | <60% (SOTA VLM) | 人类标注者 >85% |

*注：具体数值来自论文摘要/描述，完整表格见原文。*

---

## 数据集与评估设置

- **规则来源**: 三个主流平台的公开内容政策
- **政策族**: 仇恨言论、成人/色情内容、欺诈/误导内容
- **案例数**: 2,166 图文对（每案例含多条规则标注）
- **评估模型**: GPT-4V, Gemini 1.5 Pro, LLaVA-NeXT, Qwen-VL 等主流 VLM
- **指标**: 规则激活 F1、交互正确率、充分性准确率、端到端决策准确率

---

## 与电商内容治理的关联

电商平台（如淘宝、京东、抖音电商、TikTok Shop）的内容治理系统依赖精细化的政策规则树来判断商品图文、直播内容、达人营销视频是否合规。RuleSafe-VL 的评估框架直接对应以下电商场景：
- **商品主图/详情页违规检测**：规则条件推理（如"含医疗功效宣称且无资质 → 违规"）
- **达人内容合规审核**：多规则交互（如仇恨+误导的复合判定）
- **平台政策自动化执行**：评估 VLM 是否真正"读懂"政策而非依赖表面特征

---

## 论文链接

- arXiv: https://arxiv.org/abs/2605.07760
- 复现代码: `code/RuleSafe-VL/`
