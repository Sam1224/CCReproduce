# UNIVID: Unified Vision-Language Model for Video Moderation

## 基本信息 / Basic Info

| Field | Detail |
|-------|--------|
| **Title** | UNIVID: Unified Vision-Language Model for Video Moderation |
| **Authors** | Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao |
| **Affiliation** | ByteDance |
| **arXiv ID** | [2606.05748](https://arxiv.org/abs/2606.05748) |
| **Submitted** | 4 June 2026 (indexed in 8 June 2026 listing) |
| **Bucket** | STRONG |
| **Code** | `code/UNIVID/` (reproduction) |

---

## 方法概述 / Method Summary

**问题背景 / Problem**: 大规模视频平台（如 TikTok/抖音）每天需要审核海量视频内容，传统方案依赖针对单一违规类别训练的 1000+ 个专项模型，工程维护成本极高，且难以应对新兴内容风险（趋势性违规）；现有多模态 VLM 虽具通用理解能力，但缺乏内容审核所需的政策感知能力。

**Story Arc**: *"千模林立，维护成本失控 → 设计 UNIVID，用单一 VLM 主干 + 政策感知字幕作为可解释中间表示，驱动整套三阶段审核流水线"*

**方法 / Method**:

UNIVID 以 LLaVA-OV 为骨架架构，采用专门针对内容审核的训练配方：

1. **Policy-Aware Caption Generation**: 给定视频与政策文本，UNIVID 生成包含违规描述的结构化字幕（Caption）。字幕同时用于：
   - 人工可核查的决策依据
   - 多任务复用的嵌入表示

2. **训练策略 (Hybrid Supervision)**:
   - Expert Annotations（人工精标高置信样本）
   - High-quality Synthetic Data（LLM/VLM 生成的合成标注）
   - 两阶段微调：先在通用 caption 数据上 SFT，再在审核特定指令数据上 instruction tune

3. **三阶段生产系统**:
   - **Risk Filter**: 利用 UNIVID 嵌入做高吞吐早期筛查，过滤明显安全内容
   - **Moderation Actor**: 两个 UNIVID 精调变体分别做 Precision 导向和 Recall 导向决策
   - **Trend Governance**: 复用缓存的 UNIVID 字幕做聚类，自动发现新兴违规趋势

**架构图**:
```
Video Input
     │
     ▼
┌─────────────────────────────────────────────┐
│  UNIVID VLM (LLaVA-OV backbone)             │
│  + Policy Document (审核政策文本)            │
│  ─────────────────────────────────────────── │
│  → Policy-Aware Caption (可解释字幕)          │
│  → Caption Embedding (向量化)                │
└─────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   ┌───────────┐      ┌──────────────────┐
   │ Risk      │      │ Trend Governance │
   │ Filter    │      │ (聚类新兴趋势)    │
   └───────────┘      └──────────────────┘
         │
         ▼
   ┌─────────────────┐
   │ Moderation Actor│
   │ (两种UNIVID变体) │
   └─────────────────┘
         │
         ▼
    Final Decision
```

---

## 故事弧 / Story Arc

- **Insufficient**: 数千个策略专项模型 → 维护成本失控；人工规则难以追踪趋势内容
- **We Design**: UNIVID，一个政策感知的 VLM，统一字幕 + 嵌入 + 决策为一体
- **Result**: 单一主干替代 1000+ 模型，线上减少 42.7% 漏判、37.0% 误判

---

## 创新分析 / Innovation

| 维度 | 分析 |
|------|------|
| **政策感知字幕** | 区别于通用 captioning，UNIVID 字幕专门对齐审核政策，兼顾解释性与可复用性 |
| **三阶段工业流水线** | 将字幕嵌入贯穿 Risk Filter、Moderation Actor、Trend Governance，统一内部表示 |
| **趋势发现** | 自动聚类字幕嵌入以发现新兴违规话题，减少策略迭代延迟 |
| **规模效应** | 替换 1000+ 单任务模型，计算资源得到释放 |
| **可行性** | ByteDance 已在生产环境部署，指标实测，可信度高 |

---

## 关键指标 / Key Metrics

| Dataset / Platform | Metric | UNIVID | Baseline |
|-------------------|--------|--------|---------|
| TikTok Production | Violation Leakage ↓ | **−42.7%** (relative) | Prior system |
| TikTok Production | Overkill Rate ↓ | **−37.0%** (relative) | Prior system |
| Engineering | Number of Models | **1 backbone** | 1,000+ policy models |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 26 | 政策感知字幕 + 三阶段工业系统是明显创新，但 VLM captioning 本身非全新思路 |
| SOTA Delta | 15 | 13 | 生产指标改善显著（-42.7% / -37.0%），但为相对值，绝对值待补充 |
| Experimental Quality | 15 | 11 | 工业级生产部署实验，但论文级别 ablation 细节未完全公开 |
| Efficiency | 10 | 9 | 替换 1000+ 模型显著降低系统复杂度和计算开销 |
| Generalization | 5 | 4 | 跨多类违规类别通用，但仍限于字节跳达内容生态 |
| Domain Relevance | 25 | 24 | 核心就是电商/内容生态违规检测 + 达人内容治理，满分级别 |
| **Total** | **100** | **87** | 强推，生产级工作，直接对标业务痛点 |

---

## 复现说明 / Reproduction Note

代码见 `code/UNIVID/`，包含：
- 简化版 UNIVID 模型（基于 LLaVA-OV 架构）
- Policy-Aware Caption 生成训练脚本
- 模拟的 Risk Filter + Moderation Actor 流水线
- Toy 数据集接口
