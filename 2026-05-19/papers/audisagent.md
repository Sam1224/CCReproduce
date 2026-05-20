# AuDisAgent: Training-Free Multimodal Controversy Detection

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | From Static Analysis to Audience Dissemination: A Training-Free Multimodal Controversy Detection Multi-Agent Framework |
| **Authors** | Zihan Ding, Ziyuan Yang, Yi Zhang |
| **Affiliation** | Sichuan University (Ding, Zhang); Nanyang Technological University (Yang) |
| **arXiv** | https://arxiv.org/abs/2605.02939 |
| **Submitted** | May 1, 2026 |
| **Domain** | Content Governance · Multimodal Controversy Detection · Multi-Agent · Training-Free · Social Video |

---

## 方法概述 / Method Summary

多模态争议性内容检测（Multimodal Controversy Detection, MCD）用于识别视频及其评论中的争议内容，是社交视频平台风险管理的关键任务。现有方法将 MCD 视为**静态表示学习**问题（直接从视频和评论中提取特征），忽视了不同受众群体对内容的多元视角和评价。

**AuDisAgent** 受真实"内容传播到不同受众"过程启发，将 MCD 重新定义为**动态传播过程**：

1. **三类筛选智能体（Screening Agents）：**
   - **Video Agent：** 从视觉维度评估视频内容的争议潜力
   - **Comment Agent：** 从文本维度分析评论情感和争议信号
   - **Interaction Agent：** 跨模态交互分析，捕捉视频与评论之间的矛盾或共振

2. **受众传播模拟：** 这三类 Agent 模拟内容被不同受众群体接收的过程，通过 LLM 推理综合多方视角，得到最终争议判断

3. **免训练（Training-Free）：** 完全基于预训练 LLM/VLM 能力，无需任务特定微调

---

## 故事弧线 / Story Arc

> 争议性视频检测被视为静态特征分类问题，无法捕捉不同受众的多元视角 → AuDisAgent 将检测重建为受众传播动态过程，三类 Agent 分别模拟视觉/文本/交互维度的受众反馈 → 免训练架构实现可解释的多视角争议检测

---

## 创新分析 / Innovation Analysis

- 将受众传播过程引入争议检测，从"结果判断"升级为"过程模拟"，符合内容争议的真实产生机制
- 免训练架构大幅降低了部署门槛，且随 LLM 能力提升自动受益
- 三 Agent 专业化分工提供了可解释性（知道"为什么争议"）
- 相比基于监督学习的方法，对新型争议话题有更好的零样本泛化能力

---

## 关键指标 / Key Metrics

（基于论文描述，具体数值待原文确认）

| Dataset | Metric | AuDisAgent vs Baselines |
|---------|--------|------------------------|
| Social Video MCD benchmark | F1 | Outperforms supervised baselines |
| — | Accuracy | Competitive with trained models |
| — | Explainability | Qualitative improvement |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 动态传播视角新颖，三 Agent 设计合理 |
| Experimental SOTA delta | 10 | 15 | 超过监督基线，但差距细节待确认 |
| Experimental quality | 10 | 15 | 免训练设计完整，ablation 设计合理 |
| Efficiency | 8 | 10 | 免训练极大降低部署成本 |
| Generalization | 4 | 5 | 适用于各类社交视频平台 |
| Domain relevance | 22 | 25 | 直接命中：内容争议检测 + 平台治理 |
| **Total** | **76** | **100** | |

