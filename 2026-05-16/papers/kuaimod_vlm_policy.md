# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform |
| **Authors** | Lu, Zhang et al. |
| **Affiliation** | Kuaishou (快手) |
| **arXiv ID** | [2504.14904](https://arxiv.org/abs/2504.14904) |
| **Venue** | ACM SIGKDD 2025 (31st), Vol. 2 |
| **Submission Date** | April 2025 |
| **Domain Tags** | `#content-moderation` `#short-video` `#VLM` `#KDD2025` `#influencer-platform` `#benchmark` |
| **Total** | **77 / 100** |

---

## 故事弧线 / Story Arc

**现有问题:** 短视频平台内容审核面临三重困境：(1) 人工审核存在主观偏差且成本极高；(2) 自动化方法效率高但内容理解能力不足，精度受限；(3) 工业级审核规范因人工更新周期长，难以跟上内容趋势演变。

**设计方案:** 提出 **KuaiMod** — 受"普通法"（案例法）启发的内容审核框架。将 VLM 转化为**动态案例型审核者**：通过**课程学习（Curriculum Training）**实现离线自适应，通过**在线反馈机制**持续精炼审核政策。同时发布了首个附带真实用户/审核员反馈的短视频内容审核基准。

---

## 方法概述 / Method Overview

### KuaiMod 框架

```
① Offline Adaptation (Curriculum Training)
   ├── Easy cases: clear policy violations → train VLM with standard supervision
   ├── Medium cases: ambiguous → VLM learns nuanced judgment
   └── Hard cases: borderline / novel → VLM learns from reviewer rationale

② Case-Based Memory (Policy Case Store)
   ├── Each past decision stored as (video_features, label, rationale)
   ├── Retrieved during inference for in-context policy guidance
   └── Acts like legal precedent in common-law jurisprudence

③ Online Policy Refinement
   ├── Collect user/reviewer feedback on deployment decisions
   ├── Update case store and retrain via continual learning
   └── Policy evolves without long annotation cycles
```

### 短视频审核基准 (Benchmark)
- 首个含**真实用户举报 + 审核员标注**的短视频内容审核数据集
- 标注维度：违规类别、违规严重程度、细粒度判断依据

---

## 关键指标 / Key Metrics

| Dataset | Metric | KuaiMod | Best Baseline | Δ |
|---------|--------|---------|---------------|---|
| KuaiMod Benchmark | Accuracy | SOTA | Standard VLM classifier | Significant |
| KuaiMod Benchmark | Fine-grained F1 | SOTA | Rule-based system | Significant |

---

## 创新性分析 / Innovation Analysis

- **"普通法"类比新颖**: 将案例法的"先例"机制引入内容审核，使政策更新更敏捷
- **课程训练设计合理**: 从易到难逐步训练 VLM 对边界案例的判断力
- **在线反馈闭环**: 审核员反馈直接驱动策略更新，减少人工周期
- **KDD 2025 工业轨道**: 经过严格同行评审，实际部署于快手平台

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | 普通法类比新颖；课程训练+在线反馈闭环设计独特 |
| Experimental SOTA Delta | 11 | 15 | KDD 录用意味着强实验；具体数字未完全可见 |
| Experimental Quality | 10 | 15 | 真实用户/审核员标注基准；KDD 双盲评审 |
| Efficiency | 7 | 10 | 案例检索比完整 VLM 推理更高效 |
| Generalization | 4 | 5 | 在线反馈使策略适应内容演变 |
| Domain Relevance | 23 | 25 | 短视频内容治理；达人平台合规审核 |
| **Total** | **77** | **100** | |
