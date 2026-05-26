# RePAIR: Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization |
| **arXiv** | [2605.18772](https://arxiv.org/abs/2605.18772) |
| **Authors** | Gongbo Zhang, Yifan Peng, Chunhua Weng |
| **Affiliations** | Columbia University |
| **Date** | 2026-04-16 (updated May 2026) |
| **Bucket** | WEAK |
| **Total** | **51 / 100** |

---

## 故事弧 / Story Arc

> **问题:** Agentic RAG系统依赖Critic代理对LLM响应进行错误评估并迭代修正，但大多数先前工作假设Critic反馈可靠，专注于规划策略，忽视了错误修正过程本身的鲁棒性——错误分类不一致和无效修正导致修正失败。
>
> **方案:** RePAIR（Response-Action Learning Paradigm for Iterative RAG Refinement）：将缺陷RAG输出直接映射到错误缓解行动计划，无需依赖预先定义的细粒度错误分类体系。通过动作层面学习而非错误标签学习，绕过分类不对齐问题。
>
> **差异:** 现有方法需要定义和维护错误类型分类体系（容易不对齐/不完整），RePAIR直接学习"响应→修正行动"的端到端映射，更灵活、更鲁棒。

---

## 方法概述 / Method Summary

```
Flawed RAG Response
       ↓
Response-Action Mapping (trained on labeled examples)
  - 不依赖错误类型标签
  - 直接学习缺陷响应到修正行动的映射
       ↓
Error-Mitigating Action Plan
  (retry retrieval / rephrase query / discard context / ...)
       ↓
Corrected RAG Response
```

---

## 关键指标 / Key Metrics

| Benchmark | Metric | RePAIR | Agentic RAG w/ taxonomy |
|-----------|--------|--------|------------------------|
| Multiple QA benchmarks | EM/F1 | +↑ | with error taxonomy |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 16 | 30 | 绕过分类体系的想法实用，但不够突破性 |
| Experimental SOTA delta | 7 | 15 | 多基准上改进；具体数值来源于Columbia团队实验 |
| Experimental quality / ablations | 8 | 15 | 合理消融 |
| Efficiency | 5 | 10 | 简化了错误分析步骤 |
| Generalization | 3 | 5 | QA任务为主 |
| Domain relevance (ecom + governance) | 12 | 25 | RAG改进可用于电商知识问答，间接相关 |
| **Total** | **51** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **电商知识问答:** RAG驱动的商品问答/政策咨询系统受益于更鲁棒的错误修正
- **内容审核RAG:** 基于RAG的规则解读系统可受益于更稳健的错误处理
