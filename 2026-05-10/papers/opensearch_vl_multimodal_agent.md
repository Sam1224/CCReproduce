# OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents |
| 作者 | Shuang Chen et al. (10 authors) |
| 机构 | 未完整披露（GitHub: shawn0728/OpenSearch-VL）|
| arXiv | https://arxiv.org/abs/2605.05185 |
| GitHub | https://github.com/shawn0728/OpenSearch-VL |
| 提交日期 | 2026-05-06 |
| 领域标签 | 多模态搜索 · Agent · RL · 数据策划 · 工具使用 |
| 桶类别 | WEAK |
| 综合评分 | **76 / 100** |

---

## 方法概述 (中文)

多模态深度搜索 Agent 需要跨多个视觉与文本工具进行长链推理，但现有开源方案在数据质量、工具多样性和训练算法上均有明显短板：数据捷径导致单步检索崩塌，工具集不完善导致感知盲区，RL 训练中致命工具失败会污染整个回合奖励。

**OpenSearch-VL** 提供三大开放组件：
1. **数据策划管道**：基于维基百科路径采样 + 模糊实体改写 + 源锚点视觉定位，构建 SearchVL-SFT-36k（监督微调）和 SearchVL-RL-8k（强化学习）两个高质量数据集。
2. **多样工具环境**：整合文本检索、图像检索、OCR、裁剪、锐化、超分辨率、透视校正等 7 类工具，统一 Agent 的主动感知与外部知识获取。
3. **Fatal-Aware GRPO 训练算法**：对工具失败后的 token 进行 mask，保留失败前有效推理的单边优势截断（one-sided advantage clamping），避免致命失败污染整体奖励估计。

---

## 故事线 (Story Arc)

> **现状不足：** 现有多模态搜索 Agent 要么数据有捷径导致单步崩塌，要么工具集单一，要么 RL 训练对工具失败处理粗暴。
>
> **我们的解法：** 开放完整配方——数据 + 工具 + 训练算法，Fatal-Aware GRPO 精准处理工具失败，在 7 项基准平均提升 10+ 分，代码/数据/模型全部开源。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | Fatal-Aware GRPO + 完整开源管道（数据/工具/训练） |
| vs. 先前工作 | Search-o1、OpenSearcher 等均为部分开放；本文提供完整可复现方案 |
| 可行性 | 已开源，GitHub 有实现 |
| 局限 | 专注视觉搜索场景，商品目录搜索需额外适配 |

---

## 关键指标

| 数据集 | 指标 | OpenSearch-VL | 开源基线 |
|--------|------|---------------|---------|
| 7 项多模态搜索基准平均 | Accuracy | +10 pts | OpenSearcher |
| BrowseComp-Plus | Accuracy | 与商业模型持平 | — |
| BRIGHT / BEIR | 综合 | 超越 sparse+dense+rerank 基线 | — |

---

## 电商/治理迁移价值

- 商品图搜图（以图搜款）：图像检索 + OCR + 文本检索联动，可直接应用
- 违规内容溯源：多工具链式验证，可适配违规证据链构建
- 达人内容核实：跨模态搜索验证商品/场景真实性

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 24 | 30 | Fatal-Aware GRPO 算法新颖，完整开源配方价值高 |
| 实验 SOTA Delta | 12 | 15 | 7 项基准 +10pts，与商业模型持平 |
| 实验质量/消融 | 13 | 15 | 消融完整，多维度验证 |
| 效率 | 7 | 10 | 工具集并行调用，但多轮推理仍有延迟 |
| 泛化性 | 4 | 5 | 多模态搜索通用 |
| 领域相关性 | 16 | 25 | 间接相关：搜索 Agent 可赋能电商商品检索/内容验证 |
| **总分** | **76** | **100** | — |

---

## 代码与数据

- GitHub: https://github.com/shawn0728/OpenSearch-VL
- 数据集: SearchVL-SFT-36k, SearchVL-RL-8k (计划发布)
