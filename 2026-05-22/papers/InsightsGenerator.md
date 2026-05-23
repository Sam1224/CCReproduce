# Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

## 基本信息 / Basic Info

| 字段 | 值 |
|------|-----|
| **Title** | Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents |
| **arXiv** | https://arxiv.org/abs/2605.21347 |
| **Authors** | Akshay Manglik et al. (Scale AI) |
| **Affiliation** | Scale AI |
| **Submitted** | May 20, 2026 (revised May 21, 2026) |
| **Domain Tags** | `LLM-agents` `trace-diagnostics` `debugging` `multi-agent` `evaluation` |
| **Code** | Not provided |
| **Bucket** | WEAK |
| **Total** | **58** |

---

## 方法概述 / Method Overview

**中文：**
**Insights Generator（IG）**是一个多智能体系统，用于对 LLM 智能体的执行轨迹语料库进行系统性诊断分析。当前 LLM agent 失败诊断主要依赖人工抽样检查少量 trace，难以发现跨 trace 的规律性模式。IG 形式化了"语料库级 trace 诊断"问题：给定一批执行 trace，系统通过提出和验证假设，产出有证据支持的自然语言洞察报告，每条洞察关联具体的 trace 证据。系统包含：(1) 分组比较模块——区分成功/失败 trace；(2) 假设生成与验证循环；(3) 证据锚定报告生成。

**English:**
Insights Generator (IG) is a multi-agent system for corpus-level diagnostics of LLM agent execution traces. Manual inspection of small trace samples misses patterns that only emerge across populations and cannot scale to production corpora (individual traces may span tens of thousands of tokens). IG formalizes corpus-level trace diagnostics: given a batch of execution traces, it proposes and tests hypotheses across the corpus to produce an evidence-backed natural-language insights report, with each insight linked to supporting trace evidence.

---

## 故事线 / Story Arc

LLM agent 生产环境中的失败诊断依赖人工抽查少量轨迹 → 规律性 bug 难以被发现 →  
IG 将诊断问题形式化为假设驱动的语料库分析 →  
多智能体系统自动提出假设、跨 trace 验证、生成有证据的洞察报告 →  
在定性（报告质量）和定量（实施洞察后的性能提升）两个维度上评估有效性。

---

## 创新性分析 / Innovation

- **问题形式化**：将 agent 诊断提升至"语料库级"是有价值的工程化贡献，解决了生产环境中规模化 debug 的实际痛点。
- **假设驱动分析**：而非直接总结，通过假设提出-验证循环确保洞察的可信度和可复现性。
- **局限**：更偏向工程工具论文，学术创新深度有限；与电商/内容推荐领域的直接关联较弱。

---

## 关键指标 / Key Metrics

| Setting | Metric | Value |
|---------|--------|-------|
| Qualitative (rubric-based) | Report quality | Demonstrated improvements |
| Quantitative | Downstream performance after implementing IG insights | Positive delta (specific numbers not retrieved) |

---

## 评分 / Scoring

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 18 / 30 | 语料库级 trace 诊断是有价值的形式化贡献；但假设-验证循环并非全新概念 |
| 实验指标 | 8 / 15 | 双维度评估，但具体数字未完全披露 |
| 实验质量 | 10 / 15 | 定性+定量双维度，系统性评估 |
| 效率 | 5 / 10 | 多 agent 系统开销用于离线分析，可接受 |
| 泛化性 | 4 / 5 | 对任意 LLM agent 系统可用 |
| 相关性 | 13 / 25 | 可用于改进电商 AI agent（搜索/推荐/审核 agent）的调试；但不直接针对电商内容治理 |
| **Total** | **58** | |
