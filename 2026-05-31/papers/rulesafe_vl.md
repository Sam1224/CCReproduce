# RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation |
| **arXiv ID** | 2605.07760 |
| **提交日期** | 2026-05-08 |
| **作者** | (未完整披露) |
| **机构** | (Content moderation research) |
| **论文链接** | https://arxiv.org/abs/2605.07760 |
| **桶** | STRONG |
| **Total** | **71** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 当前 VLM 内容审核评测通常只测"判定结果"（有害/无害），而忽略了**推理链合理性**——模型可能"答对但逻辑错误"。RuleSafe-VL 提出基于 93 条原子规则 + 92 个规则关系的审核诊断框架，将内容审核拆解为**规则条件决策链**（Rule-Conditioned Decision Chain）：识别激活规则 → 恢复规则交互 → 判断决策充分性 → 解析最终结论。

**框架结构：**
```
平台内容政策
    ↓ 分解
93 Atomic Rules + 92 Typed Rule Relations
    ↓
内容输入（图片 + 文本）
    ↓
诊断任务链（Rule-Conditioned Decision Chain）:
  Task 1: 规则激活识别（哪些规则被触发）
  Task 2: 规则交互恢复（规则间 AND/OR/Exception 关系）
  Task 3: 决策充分性判断（证据是否足以定论）
  Task 4: 最终结论解析（Allowed / Prohibited / Restricted）
```

**与前工作差异：**
- 传统评测：分类准确率（end-to-end）
- RuleSafe-VL：分步诊断，暴露模型在规则推理中的具体弱点

---

## 关键指标 / Key Metrics

| 测试维度 | 规则数 | 关系数 | 模型表现 |
|---------|--------|--------|---------|
| 原子规则识别 | 93 | — | 多数 VLM 在复杂规则下准确率下降 |
| 规则关系推理 | — | 92 | 跨规则交互理解是当前 VLM 的主要短板 |
| 决策充分性 | 全链 | — | 中等性能，有提升空间 |

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 21 | 规则条件决策链评测框架系统性强；原子规则拆解思路新颖 |
| SOTA Δ (max 15) | 9 | Benchmark 论文，评测现有 VLM 不是提出新模型 |
| Experimental Quality (max 15) | 11 | 93 规则 + 92 关系，诊断任务链设计严密 |
| Efficiency (max 10) | 6 | 评测框架，无效率优化 |
| Generalization (max 5) | 4 | 规则框架可迁移到不同平台政策 |
| Domain Relevance (max 25) | 20 | **内容合规审核**，电商平台规则落地的关键能力 |
| **Total** | **71** | — |

---

## 创新性分析

1. **规则分解的系统性**：将复杂平台政策拆解为 93 个原子规则和类型化规则关系，使审核决策可追溯、可解释。
2. **诊断 vs 判定**：不只看模型最终答案对错，而是诊断"推理链"哪个环节失败——这对平台方提升模型可靠性更具实操价值。
3. **VLM 短板暴露**：实验表明跨规则交互推理（如"规则 A AND NOT 规则 B"）是当前 VLM 的主要弱项，为后续研究指明方向。

---

## 电商 / 达人治理落地思路

- 将电商广告法/平台规则拆解为原子规则库
- RuleSafe-VL 框架评估当前内审模型的规则推理能力
- 基于诊断结果针对性进行 SFT/RLHF 强化薄弱规则的推理
