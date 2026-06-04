# SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories |
| **Authors** | Not fully retrieved |
| **Affiliation** | Not fully retrieved |
| **arXiv** | https://arxiv.org/abs/2606.01311 |
| **HuggingFace** | https://huggingface.co/papers/2606.01311 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | LLM Agents · Skill Adaptation · Training-free · WebShop |
| **Code** | — |

---

## 方法概述 / Method Overview

### 问题 / Problem
LLM agent 越来越依赖可复用的外部技能（skills）来解决长视野交互任务。现有**无训练**技能适配流程通常从完整轨迹或会话级反馈更新技能，导致**失败归因粗粒度**（coarse failure attribution）——无法精确定位哪个步骤、哪个技能导致了失败——从而产生不稳定或过于宽泛的技能修订。

LLM agents rely on reusable external skills for long-horizon tasks. Existing **training-free** skill adaptation pipelines update skills from full trajectories, causing **coarse failure attribution** — cannot pinpoint which step or skill caused failure — leading to unstable or overly broad skill revisions.

### 方法 / Method

**SkillAdaptor：步骤级失败归因框架（Step-level Failure Attribution）**

1. **第一可操作失败步骤识别：** 给定失败轨迹，自动定位第一个可归因于技能错误的步骤（不是环境错误或 LLM 推理错误）。

2. **技能候选关联：** 将识别出的失败步骤与对应候选技能建立责任链接（responsibility attribution）。

3. **靶向更新 + 验收检查：** 对技能进行最小化精准修改，并在修改前后通过明确的验收条件（acceptance checks）验证改进，同时保持 backbone frozen。

**Story Arc:** "粗粒度轨迹反馈导致技能更新不稳定 → 步骤级失败归因实现精准靶向技能修订"

*Coarse trajectory feedback causes unstable skill updates → step-level failure attribution enables precise targeted skill revision.*

---

## 关键指标 / Key Metrics

| Benchmark | Metric | Kimi-K2.5 Gain | GLM-5 Gain | GPT-5.2 Gain |
|-----------|--------|---------------|-----------|-------------|
| PinchBench | Avg Score% | +1.5 | — | — |
| Claw-Eval | Avg Score | +1.8 | — | — |
| WebShop | Success Rate | +1.7 | — | — |

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 21 | Novel step-level failure attribution; explicit acceptance checks; responsibility linking |
| SOTA Delta | 15 | 9 | Small but consistent improvements across 3 benchmarks and multiple LLM backbones |
| Exp. Quality | 15 | 10 | Multiple benchmarks + multiple backbone LLMs |
| Efficiency | 10 | 8 | Training-free; backbone frozen |
| Generalization | 5 | 4 | Works with Kimi-K2.5, GLM-5, GPT-5.2 |
| Domain Relevance | 25 | 10 | LLM agents generally applicable; WebShop benchmark has e-commerce relevance |
| **Total** | **100** | **62** | |

---

## 电商相关性 / E-commerce Relevance

WebShop 是电商导航 agent 的标准评测场景，SkillAdaptor 在 WebShop 上 +1.7 的成功率提升直接体现了对电商 agent（自动导购、订单处理等）的价值。步骤级失败归因对电商自动化流程（下单、退款、客服 agent）的 debug 和优化也有参考意义。
