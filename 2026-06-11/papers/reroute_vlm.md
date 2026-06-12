# Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models |
| **Authors** | Cheng-Yu Yang, Shao-Yuan Lo, Yu-Lun Liu |
| **Affiliation** | (Taiwan/Academic) |
| **ArXiv** | [2606.12412](https://arxiv.org/abs/2606.12412) |
| **Submitted** | **June 11, 2026** |
| **Code** | https://github.com/elmma/mllm-reroute |
| **Domain Tags** | `VLM` `efficiency` `token-routing` `MLLM` `visual-tokens` |
| **Total** | **59 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 现有 VLM 视觉 token 剪枝方法（如 token pruning）永久移除 token，可能丢失对后续任务重要的信息，且不可恢复。

**解决方案：** 提出可恢复的视觉 token 路由机制（Recoverable Visual Token Routing），根据任务需求动态路由而非永久移除 token，在保持效率的同时保留信息可恢复性。

---

## 方法概述 / Method

- **Routing vs. Removal：** 被"剪枝"的 token 实际上被路由到辅助路径而非丢弃，在需要时可被恢复
- **Task-Adaptive Routing：** 根据下游任务的信息需求动态决定路由策略
- **Efficiency：** 在常规推理路径中减少 token 计算量，保持与剪枝方法相当的效率

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| VQA benchmarks | Accuracy vs. Pruning | Better with comparable efficiency |
| Inference | FLOPS reduction | Competitive with token pruning |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 18 | 30 | Routing concept avoids information loss vs. pruning; novel framing |
| Experimental SOTA delta | 10 | 15 | Competitive results on VQA benchmarks |
| Experimental quality / ablations | 10 | 15 | Standard CV/VLM evaluation suite |
| Efficiency | 9 | 10 | Core efficiency contribution |
| Generalization | 4 | 5 | Applicable to various VLM architectures |
| Domain relevance | 8 | 25 | VLM efficiency — indirectly relevant to deployment of content understanding models |
| **Total** | **59** | **100** | |

---

## 中文摘要

来自 June 11, 2026 的 VLM 效率工作。现有 token 剪枝方法永久移除视觉 token，可能导致不可恢复的信息损失。本文提出可恢复路由机制（Recoverable Visual Token Routing）：被"剪枝"的 token 被路由到辅助路径而非丢弃，在需要时可恢复。该方法在 VQA 等基准上优于 token 剪枝方法，同时保持相当的推理效率，为电商内容理解 VLM 的高效部署提供参考。
