# Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe |
| 作者 | Wenjin Hou et al. (16 authors) |
| 机构 | 清华大学 THUNLP 等 |
| arXiv | https://arxiv.org/abs/2605.03677 |
| GitHub | https://github.com/thunlp/OPD |
| 提交日期 | 2026-05-05 |
| 领域标签 | 知识蒸馏 · On-Policy · LLM · MLLM · 跨模态 |
| 桶类别 | WEAK |
| 综合评分 | **72 / 100** |

---

## 方法概述 (中文)

On-Policy Distillation（OPD）是将专家教师模型知识迁移给学生模型的有效范式，但在 LLM 和 MLLM 中存在两大瓶颈：① 学生采样的状态缺乏多样性（探索不足），② 教师对学生生成轨迹的监督信号不可靠（教师与学生分布偏差导致 token 级别引导失真）。

**Uni-OPD** 提出双视角优化策略：
- **学生视角**：两种数据平衡策略（难度课程采样 + 类别平衡），促进学生探索高信息量状态
- **教师视角**：基于聚合 token 级别引导与结果奖励的排序一致性分析，提出 **Outcome-Guided Margin Calibration（OGMC）** 机制，恢复正确轨迹与错误轨迹之间的序一致性

统一框架同时适用于 LLM（文本任务）和 MLLM（多模态任务），支持单教师/多教师、强弱蒸馏和跨模态蒸馏。

---

## 故事线 (Story Arc)

> **现状不足：** OPD 在 LLM 上有效，但直接迁移到 MLLM 时因教师监督不可靠和学生探索不足而性能欠佳；且无统一框架覆盖两类模型。
>
> **我们的解法：** 双视角（学生端数据平衡 + 教师端 OGMC 校准），统一 OPD 框架在 5 领域 16 基准上全面验证。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | OGMC 教师可靠性校准 + 学生双平衡策略，统一 LLM/MLLM OPD |
| vs. 先前工作 | DistiLLM、MiniLLM 等专注 LLM；无统一跨模态 OPD 方案 |
| 可行性 | THUNLP 强实验室背书，代码开源 |
| 局限 | 对超大规模模型（>70B）计算开销大 |

---

## 关键指标

| 领域 | 基准数量 | 提升 |
|------|----------|------|
| LLM 推理/QA | 8 | 优于 DistiLLM/MiniLLM |
| MLLM 多模态 | 8 | SOTA for cross-modal distillation |
| 跨模态蒸馏 | 专项 | 首个验证 |

---

## 电商/治理迁移价值

- 将大规模内容审核 VLM 蒸馏为轻量线上模型，支持实时违规检测
- 将 GPT-4o 标注能力蒸馏为小模型，降低 AIGC 内容检测成本

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 22 | 30 | OGMC 校准机制新颖；统一框架有价值 |
| 实验 SOTA Delta | 12 | 15 | 5 领域 16 基准，系统性验证 |
| 实验质量/消融 | 13 | 15 | 消融充分，多设置验证 |
| 效率 | 8 | 10 | 蒸馏本身提升效率；训练成本另算 |
| 泛化性 | 5 | 5 | LLM+MLLM 通用 |
| 领域相关性 | 12 | 25 | 间接：蒸馏技术可赋能内容理解模型压缩 |
| **总分** | **72** | **100** | — |

---

## 代码与数据

- GitHub: https://github.com/thunlp/OPD
