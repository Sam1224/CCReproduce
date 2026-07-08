# MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification |
| **作者** | Xiaotian Fan, Hiok Hian Ong, David Yuchen Wang, Zirui Zhu, Kanchan Sarkar, Kun Xu |
| **机构** | TikTok |
| **arXiv** | https://arxiv.org/abs/2606.14786 |
| **发表日期** | 2026-06-17 |
| **标签** | Reproduced Content · MLLM Distillation · Video-Audio-Text · Content Governance · Production |

---

## 故事弧 / Story Arc

**现有困境：** 搬运内容（Reproduced Content）的识别对内容平台治理至关重要，但现有方法（像素哈希、音频指纹）面对剪辑、配音替换、字幕改写等规避手段效果有限；直接用大型 MLLM 推理成本极高，无法满足亿级平台的实时审核需求。

**我们的设计：** MatchLM2Lite 用"高容量 MLLM 教师 + 轻量学生蒸馏"两阶段框架，在维持强判别性能的同时将推理成本降低 35×。

---

## 方法摘要 / Method Overview

### 系统架构

```
[Video pair]
     ↓
[MatchLM - MLLM Teacher]
  • Video encoder (visual tokens)
  • Audio encoder
  • Text encoder (metadata, captions)
  • Pairwise semantic embedding extraction
  • Fine-grained reproduction score
     ↓ Knowledge Distillation
[MatchLite - Lightweight Student]
  • Compact 3-modal fusion
  • 35× cheaper inference
  • +6.55 F1 over prior production model
```

### 关键设计决策

1. **MLLM 作为语义提取器（非生成器）**
   - 不做 next-token prediction
   - 直接利用富语义 embedding 做判别分类
   - 蒸馏目标更清晰：配对视频语义相似度

2. **三模态联合建模**
   - 视频（帧级视觉特征）
   - 音频（声学特征）
   - 文本（字幕、标题、元数据）

3. **两阶段训练**
   - Stage 1：训练 MatchLM，确立性能上界
   - Stage 2：MatchLM → MatchLite 知识蒸馏

---

## 与先前工作的差异 / Novelty vs Prior Work

| 方法 | 检测粒度 | 规避鲁棒性 | 推理成本 |
|------|----------|------------|---------|
| 感知哈希 | 像素级 | 低（易规避） | 极低 |
| 音频指纹 | 音频级 | 中（配音可规避）| 低 |
| 通用 VLM 相似度 | 语义级 | 中 | 高 |
| **MatchLM2Lite** | **语义级（三模态）** | **高** | **中（35× 优化）** |

---

## 关键指标 / Key Metrics

| 阶段 | F1-score 相对前代生产模型 | 推理成本 |
|------|--------------------------|---------|
| MatchLM（教师） | +8.57 | 高（基线） |
| MatchLite（学生）| +6.55 | 低（35× 削减） |

---

## 评分 / Score

| 维度 | 得分 | 最高 |
|------|------|------|
| Innovation | 22 | 30 |
| Experimental SOTA delta | 11 | 15 |
| Experimental quality / ablations | 12 | 15 |
| Efficiency | 9 | 10 |
| Generalization | 3 | 5 |
| Domain relevance (ecom + governance) | 23 | 25 |
| **Total** | **80** | **100** |

**评分理由：** 直接面向达人/内容平台搬运内容治理；三模态联合建模与35×成本削减结合是核心工程贡献；任务专一性（RCI）是泛化扣分原因；代码未全开放。

---

## 代码复现 / Code Reproduction

位置：`code/MatchLM2Lite/`（`2026-07-07/MatchLM2Lite/`）

复现了 MatchLM（三模态 MLLM 语义提取器）和 MatchLite（轻量蒸馏学生）的完整 pipeline，包括数据接口、模型架构、训练脚本和评估脚本。
