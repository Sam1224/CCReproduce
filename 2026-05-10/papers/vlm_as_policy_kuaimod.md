# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform |
| 作者 | Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, Kaiyu Tang, Bin Wen, Hai-Tao Zheng, Fan Yang, Tingting Gao, Di Zhang, Kun Gai |
| 机构 | 快手 (Kuaishou Technology) |
| arXiv | https://arxiv.org/abs/2504.14904 |
| 提交日期 | 2025-04-21 (v1) |
| 领域标签 | 内容审核 · 短视频平台 · VLM · CoT · 达人治理 |
| 桶类别 | STRONG |
| 综合评分 | **83 / 100** |

---

## 方法概述 (中文)

现有内容审核方法依赖静态规则分类器，当平台政策频繁更新、新型违规类型不断涌现时，规则难以及时跟进，且缺乏细粒度的推理解释能力，导致误判率高、运营人员负担重。

**KuaiMod** 提出"普通法（Common-Law）"类比框架：将 VLM 训练为基于案例推理的动态策略执行者。核心设计分三步：
1. **数据构建**：从快手真实平台抽取正负违规样本，由资深审核员结合政策文档标注，利用 GPT-4o/Qwen-VL 生成具备详细理由+最终裁决的 Chain-of-Thought（CoT）训练数据。
2. **课程训练**：先用分类任务预热，再以 CoT 推理对 VLM 进行微调，使其内化策略为"案例判决能力"。
3. **在线反馈精化**：部署后收集用户举报与审核员反馈，持续精化策略——类似普通法判例积累。

首次构建 SVP (Short Video Platform) 内容审核基准，包含真实用户/审核员反馈。

---

## 故事线 (Story Arc)

> **现状不足：** 短视频平台（如快手）内容量巨大，人工审核无法覆盖，传统分类器依赖固化规则，无法解释裁决逻辑，策略更新滞后，用户举报居高不下。
>
> **我们的解法：** 以 VLM + CoT 构建"活策略（living policy）"审核器，将政策规则内化为推理能力，通过课程训练和在线反馈实现自适应策略更新——KuaiMod。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 首次将 Common-Law "判例推理" 范式引入内容审核；VLM-as-Policy 架构 |
| vs. 先前工作 | 先前研究多用静态分类器或简单 prompt engineering；本文构建完整 CoT 数据链路 + 在线策略精化闭环 |
| 可行性 | 已在快手生产环境部署，结果实测可验证 |
| 局限 | CoT 数据构建依赖人工专家审核员，成本较高；跨平台迁移性未充分探讨 |

---

## 关键指标

| 数据集 | 指标 | KuaiMod | 基线 |
|--------|------|---------|------|
| SVP Benchmark (构建) | 用户举报率 | -20% (生产) | — |
| Kuaishou 推荐场景 | DAU | ↑ 提升显著 | baseline |
| Kuaishou 推荐场景 | APP使用时长 (AUT) | ↑ 提升显著 | baseline |
| SVP Benchmark | 判断准确率 | SOTA (细节内部) | 传统分类器 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 24 | 30 | Common-Law 范式独创，VLM-as-Policy 闭环新颖 |
| 实验 SOTA Delta | 12 | 15 | 生产 -20% 举报率 + DAU/AUT 提升，实测有力 |
| 实验质量/消融 | 13 | 15 | SVP Benchmark 构建，在线部署验证 |
| 效率 | 5 | 10 | VLM 推理较重，实时性未详述 |
| 泛化性 | 4 | 5 | SVP 场景专注，跨平台需要额外实验 |
| 领域相关性 | 25 | 25 | 直接对应内容违规检测 + 达人/UGC 治理 |
| **总分** | **83** | **100** | — |

---

## 代码与数据

- 未公开代码仓库（生产系统）
- 数据集（SVP Benchmark）：内部，未公开
- **本报告提供完整 PyTorch 复现：** `code/KuaiMod/`
