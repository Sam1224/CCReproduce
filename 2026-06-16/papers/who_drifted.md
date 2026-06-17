# Who Drifted: System or Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines |
| **作者** | Yitao Li |
| **机构** | (待确认) |
| **链接** | https://arxiv.org/abs/2606.15474 |
| **arXiv ID** | 2606.15474 |
| **提交日期** | June 13, 2026 ★ (在June 16 Monday arXiv listing中) |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**中文：**  
在使用LLM评判器持续监控产品质量时，存在一个根本性的歧义：当评分出现漂移时，是产品本身变差了，还是评判器本身被更新了？本文提出了一个统计严格的归因方法——通过固定的人工标注锚集（Anchor Set），构建双重e-process（赌注过程）：一个监控主系统漂移，另一个监控评判器-人工标注差距，并通过守卫窗口（Guard-Window）规则返回归因结论：{无漂移、系统漂移、评判器漂移}。方法提供任意时刻有效性（Anytime Validity）证明。

**English:**  
When using an LLM judge for continuous product quality monitoring, every drift alarm is ambiguous: did the product worsen or did the judge change? This paper resolves the ambiguity via a fixed human-labeled anchor set, two sequential betting e-processes (one on main stream, one on judge-vs-human gap), and a guard-window rule returning verdict ∈ {none, system, judge}. Provides theoretical anytime-validity guarantees.

---

## 故事弧线 / Story Arc

**现有方法不足 →** LLM评判器在生产环境中会被静默更新，导致评分漂移原因不明（系统退化 vs 评判器变化）。  
**本文设计 →** 双重e-process + 锚集机制，以统计严格方式实时区分漂移来源，支持持续质量监控。

---

## 创新性 / Innovation

1. **双重e-process归因框架**：统计严格的任意时刻有效性漂移归因，首次解决评判器漂移歧义。
2. **锚集设计**：固定人工标注锚集作为评判器质量的参照系，成本低且可靠。
3. **实际意义**：对生产环境中依赖LLM评判器的数据标注流水线（如内容审核质量评估）有直接价值。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 16 | 30 | 统计方法创新，实际问题重要 |
| 实验SOTA增量 (SOTA Delta) | 8 | 15 | 理论证明为主 |
| 实验质量/消融 (Exp Quality) | 10 | 15 | 模拟实验充分 |
| 效率 (Efficiency) | 7 | 10 | 锚集方案轻量 |
| 泛化性 (Generalization) | 4 | 5 | 通用于任何LLM评判器监控场景 |
| 领域相关性 (Domain Relevance) | 13 | 25 | 与内容审核质量控制相关 |
| **Total** | **58** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.15474
