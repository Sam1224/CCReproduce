# DeScore: Decoupled Reasoning and Scoring for Video Reward Modeling

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling |
| 作者 | Yuan Wang, Ouxiang Li, Yulong Xu, Borui Liao, Jiajun Liang, Jinghan Li, Meng Wang, Xintao Wang, Pengfei Wang, Kuien Liu, Xiang Wang |
| 机构 | Anonymous / multi-affiliation |
| arXiv | https://arxiv.org/abs/2605.05922 |
| 提交日期 | 2026-05-07 |
| 领域标签 | 视频奖励模型 · CoT 推理 · 多模态 · 内容质量 · AIGV |
| 桶类别 | STRONG |
| 综合评分 | **77 / 100** |

---

## 方法概述 (中文)

视频奖励模型在内容平台用得很广（推荐打分、AIGV 检测、达人内容质量审核），但现有判别式方法常常"走捷径"——不做显式推理就给分；而生成式 CoT 方案则训练不稳定，因为推理生成和打分目标耦合在一起。

**DeScore（Think, then Score）** 解耦：先用多模态语言模型生成推理（"为什么这段视频好/差"），再让独立的打分模块基于推理输出奖励。训练两阶段：
1. **判别式预热**：带 masking 提升鲁棒性
2. **RL 精调**：同时优化推理质量与奖励校准

---

## 故事线 (Story Arc)

> **现状不足：** 视频 reward model：判别式无解释、捷径学习；CoT 式生成+打分耦合训练不稳。
>
> **我们的解法：** 把"想"和"打分"解耦——多模态 LM 先生成 CoT 推理，再由独立 head 出分；两阶段训练（masked 判别 → RL 精调）。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 推理-打分解耦的两阶段架构 |
| vs. 先前工作 | 端到端 CoT 视频奖励模型（如 VL-RewardBench 路线）vs. 解耦避免训练耦合 |
| 可行性 | 标准多模态 LM + RL trainer 即可实现 |

---

## 关键指标

| 数据集 | 指标 | 备注 |
|--------|------|------|
| 视频质量打分 benchmark | 校准 + 推理一致性 | 论文中报告对照 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 24 | 30 | Think-then-Score 解耦范式 |
| 实验 SOTA delta | 11 | 15 | 论文未公开具体数字 |
| 实验质量/消融 | 12 | 15 | 双阶段训练，需 ablation |
| 效率 | 6 | 10 | 多模态 LM 推理代价 |
| 泛化性 | 4 | 5 | 视频域，部分可迁移图像/文本 |
| 领域相关性 | 20 | 25 | 视频内容打分直接服务达人/电商视频治理 |
| **Total** | **77** | **100** | |

---

## 代码与数据

- 论文未公开仓库
- **本仓库提供完整 PyTorch 复现：** `code/DeScore/`
