# Context-Driven Incremental Compression for Multi-Turn Dialogue Generation

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | Context-Driven Incremental Compression for Multi-Turn Dialogue Generation |
| **Authors** | Yeongseo Jung, Jaehyeok Kim, Eunseo Jung, Jiachuan Wang, Yongqi Zhang, Ka Chun Cheung, Simon See, Lei Chen |
| **Affiliation** | Multi-institution (HKUST, NVIDIA, etc.) |
| **ArXiv** | [2606.12411](https://arxiv.org/abs/2606.12411) |
| **Submitted** | **June 11, 2026** |
| **Conference** | ICML 2026 |
| **Domain Tags** | `LLM` `dialogue` `compression` `efficiency` `memory` |
| **Total** | **58 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 多轮对话 Agent 在每轮对话时需要重新编码不断增长的历史上下文，产生冗余的重编码和注意力计算开销，随对话长度线性增长；简单截断或摘要会损失保真度。

**解决方案：** C-DIC（Context-Driven Incremental Compression）将对话视为交织的上下文线程（contextual threads），在单个紧凑的对话记忆中存储可修订的逐线程压缩状态，通过轻量级 retrieve→revise→write-back 循环实现跨轮信息共享和过期记忆修正。

---

## 方法概述 / Method

1. **Thread-based Memory：** 将对话分解为语义相关的上下文线程，每个线程维护独立的压缩状态
2. **Retrieve→Revise→Write-back Loop：** 每轮对话：
   - Retrieve：检索相关历史线程压缩状态
   - Revise：根据当前轮更新压缩表示
   - Write-back：更新记忆中的线程状态
3. **Cross-turn Memory Sharing：** 线程状态跨轮共享，避免重复计算

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| Multi-turn dialogue | Memory efficiency | Significant reduction |
| Long conversation | Fidelity | Better than truncation/summarization |
| ICML 2026 | Acceptance | Published |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 16 | 30 | Thread-based incremental compression is novel; retrieve-revise-write-back is elegant |
| Experimental SOTA delta | 9 | 15 | ICML 2026 accepted |
| Experimental quality / ablations | 11 | 15 | ICML standard; ablations on compression components |
| Efficiency | 8 | 10 | Core efficiency focus; memory reduction is key contribution |
| Generalization | 4 | 5 | Applicable to various conversational systems |
| Domain relevance | 10 | 25 | Relevant for e-commerce customer service agents and dialogue systems |
| **Total** | **58** | **100** | |

---

## 中文摘要

来自 June 11, 2026 的 ICML 2026 工作。多轮对话 Agent 每轮需重新编码增长的历史上下文，产生冗余计算开销。C-DIC 将对话分解为语义线程，在紧凑记忆中维护可修订的逐线程压缩状态，通过 retrieve→revise→write-back 循环实现跨轮记忆共享和过期状态修正。对于电商客服 Agent 等长对话场景，可大幅降低推理成本同时保持对话保真度。
