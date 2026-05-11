## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection |
| **arXiv ID** | [2605.01638](https://arxiv.org/abs/2605.01638) |
| **提交日期** | 2026-05（exact date TBD from abstract page） |
| **作者** | Tianxiao Li, Zhenglin Huang, Haiquan Wen, Yiwei He, Xinze Li, Bingyu Zhu, Wuhui Duan, Congang Chen, Zeyu Fu, Yi Dong, Baoyuan Wu, Jason Li, Guangliang Cheng |
| **机构** | University of Liverpool · University of Exeter · The Chinese University of Hong Kong (Shenzhen) · Nanyang Technological University |
| **领域** | Deepfake Detection · Content Governance · Multimodal Benchmark · Social Media |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

Omni-Fake 是首个统一跨模态社交媒体深度伪造（Deepfake）检测基准，解决现有基准单模态、操作简单、分布不真实的三大缺陷：

1. **Omni-Fake-Set**：100 万+ 样本的大规模高质量数据集，覆盖四种模态（图像、音频、视频、音视频 Talking Head），模拟真实社交媒体分布。
2. **Omni-Fake-OOD**：20 万+ 样本的分布外测试集，专用于评估模型泛化性（刻意排除于训练集之外）。
3. **联合协议**：支持检测（是否伪造）+ 定位（哪个区域/片段被篡改）+ 解释（为什么判定为伪造）三任务联合评估。
4. **Omni-Fake-R1 检测器**：基于 Qwen2.5-Omni-7B 构建，采用四阶段重放式课程学习，结合 SFT 与 Group Sequence Policy Optimization（GSPO，RL 策略优化），自适应融合视觉与音频线索。

### Story Arc
> **现有深度伪造基准局限于单模态、分布简单** → Omni-Fake 构建涵盖四模态 100 万+ 样本的统一基准，并提出基于 RL 的 Omni-Fake-R1，自适应整合跨模态线索，在检测-定位-解释联合任务上大幅领先。

---

## 创新性分析 / Innovation Analysis

**与先前工作对比：**
- 现有基准（FaceForensics++、DeepFakeBench 等）：单模态（仅视频面部），操作单一，无法反映社交媒体真实分布
- 本文：四模态统一基准 + OOD 测试 + 联合三任务协议 + RL 驱动检测器

**关键创新：**
1. 首个统一覆盖图像/音频/视频/Talking Head 的社交媒体深度伪造基准
2. Omni-Fake-R1 使用 GSPO（组序列策略优化）将强化学习引入多模态深度伪造检测
3. OOD 基准专门测试泛化性，更贴近电商/社交平台真实违规检测场景

**可行性：** 高。数据集规模充分，RL 训练方法在 7B 量级模型上可行。

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | Omni-Fake-R1 | 说明 |
|--------|------|-------------|------|
| Omni-Fake-Set (4 modalities) | Detection AUC | SOTA | 超越所有 single-modal detector |
| Omni-Fake-OOD | Generalization AUC | SOTA | 分布外泛化最优 |
| Joint Det+Loc+Exp | Joint Score | SOTA | 首次统一三任务评测 |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 22/30 | 四模态统一基准+GSPO RL 检测器，创新显著 |
| SOTA Delta | 11/15 | 全面超越 baseline，但 Talking Head 细分数字未完全公开 |
| Exp Quality / Ablations | 12/15 | OOD 测试+联合协议，实验设计严谨 |
| Efficiency | 6/10 | Qwen2.5-Omni-7B 骨干相对轻量，但 RL 训练成本较高 |
| Generalization | 4/5 | OOD 专项测试，泛化性验证充分 |
| Domain Relevance | 21/25 | 社交媒体内容真实性治理，直接适用电商达人内容监管 |
| **总分** | **76/100** | Feishu 推送 |
