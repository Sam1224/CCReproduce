# Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning |
| **arXiv ID** | 2605.14054 |
| **Submitted** | 2026-05-13 (indexed 2026-05-14) |
| **Link** | https://arxiv.org/abs/2605.14054 |
| **Authors** | Haozhe Wang, Qixin Xu, Changpeng Wang, Taofeng Xue, Chong Peng, Wenhu Chen, Fangzhen Lin |
| **Affiliation** | (HKUST / Waterloo area, details from paper) |
| **Venue** | ICML 2026 **Spotlight** |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**问题（Story Arc）：** VLM 推理失败时，难以区分根本原因是视觉感知错误（"看错了"）还是推理逻辑缺陷（"想错了"）。现有 RL-from-feedback 框架仅对最终答案给出奖励，无法对感知过程提供显式监督，导致感知模块进步缓慢。

**解决方案：**  
- 引入**感知保真度奖励（Perception Fidelity Reward）**：在 RL 训练过程中，单独对视觉感知阶段（图像描述/localization）打分，与最终答案奖励解耦，实现感知-推理协同优化。
- 设计一个"两步评价器"：先检查视觉 grounding 是否正确（感知分），再检查基于正确感知的推理是否正确（推理分），分别反向传播奖励信号。
- 在 RLVR（RL with verifiable rewards）框架下，通过课程设计逐步提升感知难度。

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| vs. 标准 RLVR (outcome-only) | 显式解耦感知和推理奖励，使感知能力可独立被训练 |
| vs. SFT 感知增强 | RL 框架下在线自适应，避免静态 SFT 数据分布偏移 |
| ICML Spotlight 质量 | 方法严谨，消融实验完整 |

---

## 关键指标 / Key Metrics

| Benchmark | Metric | Ours | Best Prior |
|-----------|--------|------|-----------|
| MMStar | Accuracy | +4.1% abs | Base RLVR |
| MathVista | Accuracy | +3.8% abs | Base RLVR |
| VQAv2 (perception-hard split) | Accuracy | +6.2% abs | Base VLM |

---

## 评分明细 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 21 | 感知/推理双奖励解耦新颖，ICML Spotlight 背书 |
| SOTA Delta | 15 | 11 | +4-6% abs 在视觉推理 benchmark 上 |
| Experimental Quality | 15 | 12 | 详细 ablation，多 benchmark 验证 |
| Efficiency | 10 | 6 | RL 训练成本高，但推理无额外开销 |
| Generalization | 5 | 4 | 多模态多任务 benchmark |
| Domain Relevance | 25 | 11 | 感知改进有助于电商商品理解/内容审核 VLM，但非直接应用 |
| **Total** | **100** | **65** | WEAK |

---

## 故事弧 / Story Arc

> "VLM 失败时感知错误与推理错误混淆，RL 框架无法定向强化感知 → 引入感知保真度奖励解耦两类错误，在 RL 训练中显式监督视觉 grounding，实现感知-推理协同提升（ICML 2026 Spotlight）。"

---

## 电商/治理迁移价值

- **商品属性识别**：VLM 用于商品图片属性抽取时，感知错误（颜色/形状误识）是主要失败模式，本方法可定向修复
- **内容违规检测**：多模态审核 VLM 的感知准确性直接影响违规判定精度
