# Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment |
| **arXiv ID** | 2601.16027 |
| **提交日期** | 2026-01-22 |
| **作者** | Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He |
| **机构** | 中国科学院计算技术研究所 (ICT, CAS) |
| **论文链接** | https://arxiv.org/abs/2601.16027 |
| **桶** | STRONG |
| **Total** | **76** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 直播平台上的恶意行为（诈骗、协同操纵）具有**跨场次重复出现**的规律——相同主播 / 相似团伙会在多个 session 中复现"剧本"。现有方法仅对单个 session 进行风险评估，无法识别跨场次的周期性恶意模式。本文提出 **CS-VAR**（Cross-Session Evidence-Aware RAG Detector），用 RAG 从历史 session 库中检索结构化行为证据，结合 LLM 推理，再通过知识蒸馏将"全局洞察"迁移到轻量小模型，实现实时部署。

**CS-VAR 架构：**
```
当前 session (行为序列)
        ↓
[检索模块] → 历史 session 向量库 → 返回 Top-K 跨场次相似场次
        ↓
[LLM 教师] 综合当前 session + 检索到的历史证据 → 结构化推理 → 风险评分 / 解释
        ↓
[知识蒸馏] LLM 推理过程 → 轻量小模型（在线部署）
        ↓
[小模型] 识别跨场次模式，做 session-level 风险判断
```

**创新点：**
1. **跨场次 RAG**：首次将 RAG 范式引入直播风险评估，历史相似 session 作为 context
2. **LLM + 蒸馏**：LLM 作为教师（理解复杂推理），小模型作为学生（在线推理效率）
3. **结构化行为证据**：session 内高频动作序列转化为结构化 token，支持语义检索

**与前工作差异：**
- 传统方法：单 session 特征 → 分类器，无法跨 session 推理
- 本文：跨 session 检索 + LLM 推理 + 蒸馏，识别"惯犯套路"

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | CS-VAR | 最佳 Baseline |
|--------|------|--------|--------------|
| 直播风险评估（真实平台数据） | AUC | state-of-the-art | — |
| 跨场次模式识别 | Precision@K | 显著提升 | — |

> 完整实验数字请见原文。

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 22 | 首个跨场次 RAG 直播风险检测；LLM→小模型蒸馏路径新颖 |
| SOTA Δ (max 15) | 10 | 真实平台数据有显著提升，但完整数字未公开 |
| Experimental Quality (max 15) | 11 | 真实平台数据，跨场次 ablation 设计合理 |
| Efficiency (max 10) | 7 | 蒸馏后小模型支持实时 |
| Generalization (max 5) | 3 | 直播场景专用，其他流式场景可迁移 |
| Domain Relevance (max 25) | 23 | **直播风险 / 达人行为治理**核心场景 |
| **Total** | **76** | — |

---

## 创新性分析

1. **"剧本重复"洞察**：诈骗团伙在多个场次中复现相同操盘模式，RAG 检索历史 session 使模型能识别这种跨时间的违规签名。
2. **LLM 推理 + 蒸馏**：LLM 负责高质量推理生成监督信号，小模型在线服务——解决了 LLM 推理延迟与生产部署的矛盾。
3. **结构化行为表示**：原始直播行为流（送礼、弹幕、购买、互动）被结构化为 session-level 向量，支持高效语义检索。

---

## 电商 / 达人治理落地思路

- 将平台历史高风险直播场次建立向量索引库
- 新直播实时检索历史相似 session → 喂给规则/LLM 做风险预判
- 结合 CS-VAR 的蒸馏小模型做秒级实时风控
