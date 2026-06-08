## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints |
| **Authors** | Jiayu Liu, Cheng Qian, Zhenhailong Wang, Bingxuan Li, Jiateng Liu, Heng Wang, Jeonghwan Kim, Yumeng Wang, Xiusi Chen, Yi R. (May) Fung, Heng Ji |
| **Affiliations** | University of Illinois Urbana-Champaign (UIUC) |
| **ArXiv ID** | [2606.05622](https://arxiv.org/abs/2606.05622) |
| **Submitted** | 2026-06-03 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.AI, cs.CL |
| **Code** | [JiayuJeff/AdaPlanBench](https://github.com/JiayuJeff/AdaPlanBench) (official) |
| **Dataset** | [HuggingFace: JiayuJeff/AdaPlanBench](https://huggingface.co/datasets/JiayuJeff/AdaPlanBench) |
| **Bucket** | WEAK→MEDIUM |
| **Total** | **80 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 25 | 30 | Dual-constraint (world + user) progressive disclosure protocol is novel; violation-triggered revelation mechanism turns constraint induction/tracking into a measurable skill |
| Experimental SOTA delta | 10 | 15 | Best model only 67.75% accuracy; demonstrates significant headroom; performance monotonically degrades with constraint accumulation |
| Experimental quality / ablations | 13 | 15 | 10 leading LLMs evaluated; separate analysis for world vs. user constraint difficulty; constraint accumulation curves |
| Efficiency | 6 | 10 | Benchmark paper; multi-turn interaction protocol adds evaluation overhead |
| Generalization | 4 | 5 | Household tasks; constraint pipeline is transferable to other domains |
| Domain relevance | 22 | 25 | Planning under progressively revealed constraints is central to governance-aware agents (compliance rules, user preference, platform policy all progressively impose constraints on action) |
| **Total** | **80** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
真实世界中的 LLM agent 必须在"世界约束"（资源可用性、环境限制）与"用户约束"（个人偏好、禁忌）同时存在的情况下规划任务。更重要的是，这些约束通常不是预先全部告知的，而是在 agent 执行过程中逐步暴露（例如用户说"我不喜欢鸡蛋"只在 agent 提出含鸡蛋的方案之后才提出）。现有规划 benchmark 要么只考虑一类约束，要么一次性给出所有约束，严重低估了真实场景的难度。

**X is insufficient → we design Y to solve it：**
> 现有规划 benchmark 将约束视为静态输入，无法评测"约束归纳"与"自适应重规划"能力。AdaPlanBench 构建了"双约束 + 渐进揭示"的多轮交互协议：agent 提出计划 → 系统检查是否违反任意约束 → 仅在发生违规时才告知该约束 → agent 在记住已积累约束的同时修订计划，循环直至计划可行或轮次耗尽。

### 核心方法

1. **双约束体系**：
   - *世界约束*：基于家庭环境资源/可用性（某物品不存在、某操作受限）
   - *用户约束*：基于用户偏好/禁忌（不想用某品牌、有饮食限制）

2. **可扩展约束构建流水线**：为 307 个 ALFRED 家庭任务自动生成双约束画像，覆盖多种约束类型组合。

3. **违约驱动的渐进揭示协议**：约束池中的约束仅在 agent 提出违规计划时才"激活"并告知，迫使 agent 在多轮中累积约束记忆并反复修订。

4. **评测指标**：最终计划的成功率（满足所有约束的可行计划生成率）+ 约束轮次积累下的性能曲线。

### English Summary

AdaPlanBench is a multi-turn interactive planning benchmark built on 307 household tasks. It augments each task with dual constraints — world constraints (resource/environment limits) and user constraints (preferences/taboos) — through a scalable pipeline. The key mechanism is violation-triggered progressive disclosure: constraints are not given upfront; they are revealed only when an agent's proposed plan violates them, forcing iterative replanning while tracking accumulated constraint feedback. Experiments on 10 leading LLMs show that even the best model achieves only 67.75% accuracy, with performance degrading significantly as more constraints accumulate, and user constraints proving harder than world constraints.

---

## 创新点分析 / Innovation Analysis

**中文：** 核心创新是将"约束归纳（constraint induction）"和"约束跟踪（constraint tracking）"从规划能力的隐含前提提升为可量化的评测维度。违约驱动的渐进揭示机制比"给定所有约束后规划"更贴近真实 agent 部署场景，且天然支持测量 agent 在约束积累下的性能退化曲线。双约束体系的统一构建方法也为其他领域（如电商购物 agent 在处理用户偏好与库存约束时）提供了可复用的框架。

**English:** The core innovation is elevating constraint induction and tracking from implicit planning prerequisites to explicit, measurable evaluation dimensions. The violation-triggered disclosure mechanism reflects real deployment conditions where constraints emerge through interaction rather than being provided upfront. The dual-constraint construction pipeline is reusable for building governance-relevant benchmarks in other domains (e.g., shopping agents navigating user preference vs. inventory constraints, compliance agents navigating policy vs. operational constraints).

---

## 关键指标 / Key Metrics

| Setting | Best Model | Accuracy | Notes |
|---------|-----------|----------|-------|
| Dual-constraint (world + user) | Best among 10 LLMs | **67.75%** | Significant headroom for improvement |
| World constraint only | — | Higher than dual | User constraints harder than world |
| Constraint accumulation | All models | Monotonically decreases | More constraints → worse performance |

---

## 代码复现 / Code Reproduction

官方代码及数据集已由作者发布：
- GitHub: [JiayuJeff/AdaPlanBench](https://github.com/JiayuJeff/AdaPlanBench)
- Dataset: [HuggingFace](https://huggingface.co/datasets/JiayuJeff/AdaPlanBench)

无需本仓库另行复现。
