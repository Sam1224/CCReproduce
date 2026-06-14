# UNIVID: Unified Vision-Language Model for Video Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | UNIVID: Unified Vision-Language Model for Video Moderation |
| **ArXiv ID** | [2606.05748](https://arxiv.org/abs/2606.05748) |
| **Authors** | Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao |
| **Affiliation** | ByteDance |
| **Submitted** | 2026-06-04 |
| **Source** | WebSearch fallback (arXiv direct 403) |
| **Bucket** | STRONG — 内容治理、视频审核、VLM |
| **Code** | `2026-06-13/code/UNIVID/` |

---

## 方法概述 / Method Overview

**故事弧线：** 工业级视频审核系统长期依赖数千个针对特定政策的黑盒分类器，这些分类器难以维护、缺乏透明度、无法跨任务复用，且在面对新政策时需从零开发。→ UNIVID 提出以单一策略感知字幕 VLM 替代碎片化分类器生态：先由 UNIVID 生成人类可验证的政策感知字幕（interpretable intermediate），再由下游轻量模型利用字幕做最终决策。

**三阶段审核 Pipeline：**
- **Stage A — Risk Filter**：多模态风险漏斗，融合 UNIVID 字幕与视觉特征，过滤掉明显低风险视频，减少后续计算压力。
- **Stage B — Moderation Actor**：两个下游微调模型并行工作：  
  - *UNIVID-Lite*：轻量级，预测初步审核决策；  
  - *UNIVID-RAG*：基于历史违规事件的检索增强，召回已有违规模式。  
- **Stage C — Trend Governance**：缓存 UNIVID 嵌入，动态检测新兴风险趋势（训练特定 trend head），无需重新标注全量数据。

**训练数据配方：** 结合人工专家精标数据（human-refined labels）与合成数据（synthetic data），在多政策、多模态场景下联合训练单一字幕生成器。

**核心创新：**
1. 以"策略感知字幕"作为可解释中间表示，使审核决策人类可读可验证；
2. 单一 UNIVID backbone 替换 1000+ 政策特定模型，实现跨任务复用；
3. UNIVID-RAG 将违规历史写入检索，减少违规漏检（violation leakage）；
4. Trend Governance 利用缓存嵌入做低成本新兴风险追踪。

---

## 关键指标 / Key Metrics

| 指标 | UNIVID Pipeline | Baseline |
|------|-----------------|---------|
| Violation Leakage (↓) | **-42.7%**（相对下降） | baseline 系统 |
| Overkill Rate (↓) | **-37.0%**（相对下降） | baseline 系统 |
| 模型维护数量 | 1（单 backbone） | **1,000+** 特定分类器 |

指标来源：ByteDance 线上工业部署对比实验

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 24 | 30 | 单 VLM 替代千模型生态；策略字幕作为可解释中间表示；三阶段 pipeline 设计精巧 |
| 实验指标 | 13 | 15 | 工业 AB：42.7% 漏检降低 + 37% 误杀降低，真实部署有说服力 |
| 实验质量 | 12 | 15 | 工业级对比，数据规模不可公开但架构 ablation 完整 |
| 方法效率 | 9 | 10 | 1000+ 模型→1，计算资源大幅释放 |
| 方法泛化性 | 4 | 5 | 字幕-审核范式可推广到任意内容平台 |
| 领域相关性 | 24 | 25 | 直接命中内容治理/视频审核/达人内容风险管理，ByteDance 即 TikTok/抖音 |
| **Total** | **86** | **100** | |

**复现路径：** `2026-06-13/code/UNIVID/`

---

## Story Arc

> **现状不足：** 全球化内容平台面临"政策碎片化 × 黑盒分类 × 缺乏透明度"三重困境；1000+ 模型维护成本极高，新政策上线周期长。  
> **解法：** 用单一策略感知 VLM（UNIVID）生成人类可验证字幕 → 三阶段 pipeline（过滤/决策/趋势治理）精细化利用字幕 → 线上验证违规漏检与误杀均大幅下降。

---

## 与先前工作的区别

- 传统方案：专用图像分类器 + 规则引擎，每政策一模型；
- CLIP/ViLT 类：多模态表示但无政策感知、无可解释输出；
- UNIVID：首次在工业部署中以"策略字幕生成"取代"直接分类"，将 1000+ 模型整合为单一骨干。
