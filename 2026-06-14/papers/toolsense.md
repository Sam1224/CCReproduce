## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs |
| **作者** | Ashutosh Hathidara, Sai Shruthi Sistla, Sebastian Schreiber, Sahil Bansal |
| **机构** | SAP Labs |
| **arXiv** | [2606.12451](https://arxiv.org/abs/2606.12451) |
| **代码** | [github.com/SAP/toolsense](https://github.com/SAP/toolsense) |
| **发布日期** | 2026-06-12 |
| **领域标签** | Agent、工具检索、评测/诊断、Benchmark、Internalization Score |

---

## 方法概述

**Story Arc**：把工具目录参数化写进 LLM（parametric tool retrieval）的方案在标准 ToolBench 上表现优异，但用的是详尽描述 + constrained decoding，可能掩盖模型是否真的"理解"工具。ToolSense 为任意工具目录自动生成三类诊断集：
- **Realistic Retrieval Benchmark (RRB)**：按三档歧义程度生成更贴近真实用户表达的 query；
- **MCQ Tool Probing**：测试对工具能力边界的细粒度理解；
- **QA Probing**：开放式工具知识问答。

同时提出 **Internalization Score**（free-form vs constrained decoding 准确率之差），量化模型对 constrained decoding trie 的依赖程度——分差越大说明模型在 free-form 情况下工具知识实际上很薄弱。

**Innovation vs Prior Work**：ToolBench/ToolEval 等基准聚焦执行成功率，但不区分"检索出工具"与"真正理解工具"。ToolSense 首次把两者解耦，并给出可自动生成的评测框架，适合生产型 Agent 上线前的风险审计。

---

## 关键指标

| 测试集 | 指标 | 现象 |
|--------|------|------|
| ToolBench (~47k tools) | Realistic Retrieval Accuracy | 部分参数化配置较标准 ToolBench 评测**下滑 50–64 个百分点** |
| ToolBench | Factual Probing Accuracy | 部分配置接近**随机水平**，揭示知识-检索解耦 |
| 多模型评测 | Internalization Score | 可量化 constrained decoding 依赖程度差异 |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 24/30 | 评测框架创新：解耦 pattern matching vs tool understanding，引入 Internalization Score |
| 实验指标 | 12/15 | 50–64pp 的下滑揭示重大评测 gap，但无直接"SOTA 提升"指标 |
| 实验质量 | 12/15 | 多模型横向对比，消融充分；以诊断为主 |
| 效率 | 7/10 | 自动生成评测集，规模可扩展 |
| 泛化性 | 4/5 | 框架对任意工具目录适用 |
| 领域相关性 | 21/25 | 对电商/治理类 Agent 工具检索质量管控非常有价值 |
| **Total** | **80/100** | 评测与诊断框架，对 Agent 工具可靠性治理有直接参考价值 |

---

## 复现

官方代码：[github.com/SAP/toolsense](https://github.com/SAP/toolsense) — 已验证。  
本仓库路径：`2026-06-14/ToolSense/`（含 verify_repo.py）
