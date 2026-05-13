# Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models |
| 作者 | Yuancheng Wei, Linli Yao, Lei Li, Haojie Zhang, Hao Zhou, Fandong Meng, Xu Sun |
| 机构 | 北京大学 (Peking University) + WeChat AI (腾讯) |
| arXiv | https://arxiv.org/abs/2605.07872 |
| 提交日期 | 2026-05-08 (arXiv 列表：2026-05-11) |
| 领域标签 | 视频理解 · Reward Model · MLLM · 偏好数据 · 基准评测 · RLHF |
| 桶类别 | WEAK |
| 综合评分 | **67 / 100** |

---

## 方法概述 (中文)

文本和图像领域的奖励模型（Reward Model）已相当成熟（InternLM2-Reward、Qwen2.5-RM 等），但视频理解的奖励建模严重滞后——缺乏专用基准和高质量偏好数据集，导致视频 MLLM 的 RLHF/RLAIF 训练受阻。

**本文提出三件套：**

### 1. VURB（Video Understanding Reward Bench）
- 2,100 个偏好对（preference pairs）
- 每对包含**长链推理（CoT）痕迹**，平均 1,143 token
- 采用**多数投票评测**（majority voting evaluation）提升标注可靠性
- 覆盖三类视频任务：通用视频理解、长视频理解、视频推理

### 2. VUP-35K（Video Understanding Preference Dataset）
- 35,000 条全自动化流程生成的高质量视频偏好数据
- 提供大规模 reward model 训练监督信号
- 完全自动化构建，无需昂贵的人工标注

### 3. VideoDRM & VideoGRM
- 基于 VURB 和 VUP-35K 训练的视频奖励模型
- 在 VURB 基准上达到 SOTA 性能

---

## 故事线 (Story Arc)

> **现状不足：** 多模态奖励模型在文本/图像领域已充分发展，但视频理解奖励建模严重受限于：(1) 缺乏带有推理痕迹的高质量评测基准；(2) 缺乏大规模视频偏好训练数据。这导致视频 MLLM 无法有效进行 RLHF 对齐，视频生成/理解质量难以量化和提升。
>
> **我们的解法：** 提出 VURB 基准（2,100 对 + CoT 痕迹 + 多数投票）、VUP-35K 自动化偏好数据集，以及基于此训练的 VideoDRM/VideoGRM，形成完整的视频奖励建模框架。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 首个带 CoT 推理痕迹的视频理解奖励基准；全自动化偏好数据构建流水线 |
| vs. 先前工作 | VideoRewardBench 等先前工作无 CoT 痕迹，标注可靠性低；VUP-35K 自动化规模远超人工标注 |
| 可行性 | 北大 + 腾讯合作，工程可行性高 |
| 局限 | 自动化标注可能引入系统偏差；视频时长和类型覆盖范围待进一步扩展 |

---

## 关键指标

| 数据集 | 指标 | 本文 | 说明 |
|--------|------|------|------|
| VURB | 偏好对数量 | 2,100 | 专家标注 + 多数投票 |
| VURB | 平均 CoT 长度 | 1,143 tokens | 长链推理痕迹 |
| VUP-35K | 训练样本量 | 35,000 | 全自动化流水线 |
| VideoDRM/VideoGRM | VURB Accuracy | SOTA | vs. 先前视频奖励模型 |
| 视频任务覆盖 | 类型 | 3 | 通用/长视频/推理视频 |

---

## 电商/内容平台应用场景

- **短视频内容质量评估**：基于奖励模型自动评估电商短视频（商品介绍、开箱视频）的质量，替代人工审核
- **直播剪辑评估**：对直播精彩片段的自动化质量打分
- **MLLM 微调反馈**：为电商平台自有视频理解模型提供 RLHF 训练信号
- **内容违规的推理链条**：有 CoT 推理的奖励模型更适合内容合规的可解释审核

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 21 | 30 | CoT 痕迹 + 自动化流水线组合新颖；但属于工程性综合工作，方法原创性适中 |
| 实验 SOTA Delta | 10 | 15 | 新基准 SOTA，但缺少与真实下游任务改进的直接对比 |
| 实验质量/消融 | 11 | 15 | 多数投票提升可靠性；三类视频任务覆盖；自动化流水线消融待看 |
| 效率 | 7 | 10 | 自动化数据构建降低成本；模型推理效率未特别讨论 |
| 泛化性 | 4 | 5 | 通用/长视频/推理三类覆盖较全，可迁移到电商视频 |
| 领域相关性 | 14 | 25 | 间接相关：视频质量评估对内容平台有价值，但非直接电商/治理工具 |
| **Total** | **67** | **100** | — |

---

## 代码与数据

- VURB 基准：待发布
- VUP-35K 数据集：待发布
- 模型权重：待发布
