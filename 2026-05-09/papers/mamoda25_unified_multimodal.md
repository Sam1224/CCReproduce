# Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE |
| 作者 | ByteDance Seed 团队（mammothmoda 项目） |
| 机构 | ByteDance (字节跳动 Seed 研究团队) |
| arXiv | https://arxiv.org/abs/2605.02641 |
| GitHub | https://github.com/bytedance/mammothmoda |
| 提交日期 | 2026-05-04 |
| 领域标签 | 统一多模态 · 文生图/视频 · 视频编辑 · DiT · MoE · 蒸馏 |
| 桶类别 | WEAK |
| 综合评分 | **67 / 100** |

---

## 方法概述 (中文)

多模态生成/理解任务通常需要部署多个专用模型（文生图模型、视频编辑模型、多模态理解模型等），系统复杂度高、参数冗余严重。

**Mamoda2.5** 提出统一 AR-Diffusion 框架，在单一模型中无缝集成多模态理解与生成：
1. **架构**：以 Diffusion Transformer（DiT）为骨干，引入细粒度混合专家（MoE）设计——128 个专家、Top-8 路由，形成 25B 总参数但仅激活 3B 参数的稀疏 MoE 模型，显著降低训练成本同时扩大模型容量。
2. **支持任务**：文生图（T2I）、文生视频（T2V）、图像编辑（Image Editing）、视频编辑（Video Editing）、多模态理解（VQA）——单模型统一支持。
3. **推理加速**：提出联合少步蒸馏（Few-Step Distillation）+强化学习框架，将 30 步编辑模型压缩为 4 步模型，大幅加速推理。
4. **视频编辑 SOTA**：在 OpenVE-Bench 视频编辑质量评测上超越所有开源模型，与 Kling O1 等顶级闭源模型持平。

---

## 故事线 (Story Arc)

> **现状不足：** 多模态内容生成场景（电商创意内容生产、达人视频制作）需要部署多个分立模型，维护成本高；Diffusion 模型推理步骤多，延迟高（30步）。
>
> **我们的解法：** 用单一 DiT-MoE 架构统一所有生成/理解任务，MoE 实现参数高效扩展，联合蒸馏+RL 将编辑模型压缩为 4 步推理——Mamoda2.5。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | AR+Diffusion 统一架构 + DiT-MoE 稀疏化 |
| vs. 先前工作 | 先前统一模型（如Janus, Show-o）通常混用 AR/Diffusion 效果参差；Mamoda2.5 用 DiT-MoE 实现更平滑的多任务统一 |
| 效率创新 | MoE 仅激活 3B/25B 参数；4步推理蒸馏（原30步） |
| 工业价值 | ByteDance 生产团队，有在字节系产品落地基础 |
| 局限 | MoE 系统通信开销在分布式训练中较高；视频编辑质量仍落后部分闭源顶尖模型 |

---

## 关键指标

| 数据集 / 任务 | 指标 | Mamoda2.5 | 对比模型 |
|--------------|------|-----------|---------|
| VBench 2.0（文生视频） | 综合评分 | 开源 Top 级 | 开源 SOTA |
| OpenVE-Bench（视频编辑） | 质量评分 | **与 Kling O1 持平** | 所有开源模型 |
| 编辑推理步数 | 步数 | **4步**（原 30步） | 蒸馏前基线 |
| 模型参数 | 激活参数 | **3B / 25B total** | 密集等效模型 |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | AR-Diffusion统一+DiT-MoE+联合蒸馏RL，三重创新 |
| Experimental SOTA delta | 11 | 15 | VBench 2.0开源SOTA，OpenVE-Bench匹配Kling O1 |
| Experimental quality | 11 | 15 | 多任务多基准，生成+理解全覆盖 |
| Efficiency | 7 | 10 | MoE激活3B/25B，4步推理 |
| Generalization | 4 | 5 | 5种多模态任务统一支持 |
| Domain relevance | 12 | 25 | 电商内容创意生成，间接相关 |
| **Total** | **67** | **100** | |
