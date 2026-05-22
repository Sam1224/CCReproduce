# CS-VAR: Deja Vu in Plots — Cross-Session RAG for Live Streaming Risk Assessment

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment |
| 作者 | Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He |
| 机构 | 中国科学院计算技术研究所 (ICT, CAS) |
| arXiv | https://arxiv.org/abs/2601.16027 |
| 提交日期 | 2026-01-22 (v1) — *注：早于 May 21 窗口，作为参考* |
| 领域标签 | 直播风控 · RAG · LLM 蒸馏 · 跨会话行为检测 · 实时部署 |
| 桶类别 | STRONG (reference only — outside May 21 window) |
| **总分 / Total** | **76 / 100** |

---

## 方法概述 (中文)

直播平台面临的高风险行为（诈骗、协同刷量、有害内容）往往具有以下特点：
- **渐进性**：风险信号在单场直播内难以直接判断；
- **跨会话复现**：同一风险剧情在不同主播、不同时间段重复出现。

**CS-VAR（Cross-Session Evidence-Aware Retrieval-Augmented Detector）** 设计架构如下：

1. **双模型协作**: 轻量级域专用检测模型（Student）负责实时会话级风险推理；LLM 教师模型负责跨会话证据推理。
2. **跨会话检索 (Cross-Session Retrieval)**: 对当前直播会话提取行为特征，检索历史数据库中语义相似的历史风险会话作为"案例证据"。
3. **LLM 推理蒸馏 (LLM Reasoning Distillation)**: LLM 对检索到的历史证据进行推理，识别跨会话的局部-全局风险模式，将这种"全局洞察"通过蒸馏迁移到轻量级 Student 模型，使其无需在线调用 LLM 即可识别跨会话模式。

---

## 故事线 (Story Arc)

> **现状不足：** 直播风控模型专注单场会话内的行为特征，但许多诈骗/协同恶意行为在单场内难以判断——关键证据散落在不同时间、不同账号的历史会话中。
>
> **我们的解法：** CS-VAR 通过跨会话检索提供历史证据，让 LLM 进行全局推理，并将这种"记忆力"蒸馏进轻量实时模型——让小模型"记得"看过的剧情。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 将 RAG 与 LLM-to-Student 蒸馏结合用于跨会话实时风险检测；跨会话证据检索思路新颖 |
| vs. 先前工作 | 先前风控模型（GNN/RNN）局限单会话特征；无跨会话检索机制 |
| 可行性 | 基于真实直播平台数据，中科院实验室背书 |
| 局限 | 检索数据库随时间增长带来延迟；蒸馏质量依赖 LLM 推理准确性 |

---

## 关键指标

| 数据集 | 指标 | CS-VAR | 基线 |
|--------|------|--------|------|
| 真实直播平台风险数据集 | F1 Score | 超越 GNN/RNN 基线 | RNN/GNN 单会话检测 |
| 真实直播平台风险数据集 | AUC | 显著提升 | 最强单会话基线 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 跨会话 RAG + LLM 蒸馏组合新颖 |
| 实验 SOTA Delta | 9 | 15 | 公开指标有限；声称超越基线但具体数值不详 |
| 实验质量/消融 | 11 | 15 | 真实平台数据；有跨会话证据消融 |
| 效率 | 9 | 10 | Student 模型实时部署，LLM 仅训练时使用 |
| 泛化性 | 3 | 5 | 仅限直播场景 |
| 领域相关性 | 22 | 25 | 直播风险检测直接对应内容治理/达人行为监控 |
| **总分 / Total** | **76** | **100** | — |

---

## 代码与数据

- arXiv: https://arxiv.org/abs/2601.16027
- 无公开代码
- *注：此论文提交于 2026-01-22，早于本次 May 21 检索窗口，作为领域参考列出。*
