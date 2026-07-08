# Bridging Short Videos and Live Streams: Reasoning-Guided Multimodal LLMs for Cross-Domain Representation Learning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Bridging Short Videos and Live Streams: Reasoning-Guided Multimodal LLMs for Cross-Domain Representation Learning |
| **作者** | Le Zhang, Xiaolan Zhu, Yuchen Wang, Jiajing Ye, Zhicheng Dou, Ruqing Zhang, Shuaiqiang Wang |
| **机构** | Kuaishou Technology; Renmin University of China |
| **arXiv** | https://arxiv.org/abs/2606.04448 |
| **发表日期** | 2026-06-03 |
| **标签** | Cross-Domain Recommendation · Live Streaming · MLLM · Distillation · Kuaishou |

---

## 故事弧 / Story Arc

**现有困境：** 直播推荐面临严重冷启问题——短视频流量巨大、行为信号丰富，而直播作为核心转化场景行为稀疏，item 直接表示极难学习。现有跨域推荐多迁移用户行为，忽略了 item 语义层面的可迁移知识。

**我们的设计：** RGCD-Rep 通过两阶段框架迁移 MLLM 推理生成的 item 语义表示：① 推理感知蒸馏让轻量学生 MLLM 习得跨域推理知识；② 可迁移性引导的表示分解把 item 表示解耦为"域无关语义"和"直播特有信息"，实现可迁移知识的精准迁移。

---

## 方法摘要 / Method Overview

### 两阶段框架

**Stage 1：推理感知蒸馏（Reasoning-Aware Distillation）**
- 冻结教师 MLLM：从短视频-直播配对中生成结构化跨域推理知识
- 轻量学生 MLLM：蒸馏教师的推理知识，在不大幅增加参数的情况下理解跨域语义

**Stage 2：可迁移性引导的跨域表示学习（Transferability-Guided CDRepL）**
- Item 表示分解：
  - $r_{transfer}$：可迁移表示（跨域共通语义，如商品属性、内容主题）
  - $r_{residual}$：领域残差（直播特有信息，如主播风格、直播状态）
- 行为协作对齐：用跨域用户行为信号引导可迁移表示的质量
- 表示离线预计算，低成本接入下游检索系统

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | RGCD-Rep | 最强 Baseline |
|--------|------|----------|-------------|
| 直播推荐离线 | Metric-1 | SOTA | -17.59% |
| 直播推荐离线 | Metric-2 | SOTA | -30.93% |
| 快手线上 A/B | 多核心业务指标 | 显著提升 | — |

---

## 评分 / Score

| 维度 | 得分 | 最高 |
|------|------|------|
| Innovation | 20 | 30 |
| Experimental SOTA delta | 13 | 15 |
| Experimental quality / ablations | 12 | 15 |
| Efficiency | 7 | 10 |
| Generalization | 3 | 5 |
| Domain relevance (ecom + governance) | 22 | 25 |
| **Total** | **77** | **100** |

**评分理由：** 快手直播电商场景高度匹配，A/B 部署验证，跨域冷启方案有实际价值。扣分原因：跨域仅限"短视频→直播"单一路径，学生 MLLM 仍有一定参数成本。
