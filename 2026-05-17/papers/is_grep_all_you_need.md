# Is Grep All You Need? How Agent Harnesses Reshape Agentic Search

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Is Grep All You Need? How Agent Harnesses Reshape Agentic Search |
| **Authors** | Sahil Sen, Akhil Kasturi, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah |
| **ArXiv** | https://arxiv.org/abs/2605.15184 |
| **Submitted** | 14 May 2026 |
| **Venue** | arXiv preprint |
| **Code** | See paper |
| **Domain** | Agentic RAG, retrieval strategy, LLM agents, search |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: RAG（检索增强生成）被广泛采用于 Agentic Search，但现有文献缺乏对检索策略选择与 Agent 架构、工具调用范式交互关系的系统对比研究。尤其是工具输出呈现方式（inline vs file-based）和搜索文本噪声对性能的影响尚未深入探讨。  
> **我们的设计**: 设计了一个双实验体系，比较 grep 搜索和向量检索（vector retrieval）在 LongMemEval 基准上的效果，测试了多种 Agent 框架（Chronos、Claude Code、Codex、Gemini CLI）和工具结果呈现方式，揭示"inline grep 在几乎所有框架上优于 inline vector"这一反直觉发现。

### 技术细节 / Technical Details

**实验设计**:
- **Experiment 1**: 116个问题（LongMemEval），比较 grep vs vector + 4种 agent 框架
- **Experiment 2**: 在存在更多干扰文本的设置下测试鲁棒性
- **工具呈现方式**: Inline（结果直接插入上下文）vs File-based（写入文件后引用）
- **Agent 框架**: Chronos（自定义）、Claude Code、Codex、Gemini CLI

**核心发现**:
- Inline grep 在每种 agent-model 组合上均优于 inline vector
- 差距有时很显著，特别是在有干扰文本时
- 文件级 grep 和 vector 检索的差距更复杂

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **反直觉发现** | Grep（精确关键词搜索）在多数设置下优于向量语义搜索，挑战了"向量检索更优"的行业假设 |
| **多框架系统测试** | 跨4个主流 Agent 框架的系统测试，结论较可靠 |
| **实用启示** | 为中等规模数据集（~1000文档）的 Agent 搜索提供实际选型建议 |
| **工具呈现方式** | 首次系统研究"工具输出如何呈现给模型"对搜索效果的影响 |

---

## 关键指标 / Key Metrics

| 评测集 | 指标 | Inline Grep | Inline Vector | 差异 |
|--------|------|-------------|---------------|------|
| LongMemEval (116Q) | Accuracy | **Higher** | Lower | Grep wins all harness pairs |
| With distractor text | Accuracy | More robust | Less robust | — |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 16 | 反直觉发现有价值；但系统对比偏工程实验，理论贡献一般 |
| Experimental SOTA delta | 15 | 8 | 实证发现而非方法创新；无传统 SOTA 对比 |
| Experimental quality/ablations | 15 | 10 | 多框架系统实验设计合理 |
| Efficiency | 10 | 6 | Grep 速度快是天然优势 |
| Generalization | 5 | 3 | 仅 LongMemEval，泛化性有限 |
| **Domain relevance** | **25** | **9** | Agentic search 与 RAG 系统相关，但与电商内容治理关联较弱 |
| **Total** | **100** | **52** | 实用价值中等，适合工程参考 |
