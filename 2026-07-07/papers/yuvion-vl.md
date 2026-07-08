# Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety |
| **作者** | Ting Cao et al. |
| **机构** | Yuvion AI |
| **arXiv** | https://arxiv.org/abs/2606.25034 |
| **发表日期** | 2026-06-23 |
| **标签** | Content Safety · MLLM · E-commerce Governance · Adversarial · C2FT |

---

## 故事弧 / Story Arc

**现有困境：** 通用多模态模型在真实内容安全场景中频繁失误——面对小字/水印规避、logo 变形、遮挡隐藏、AI 生成内容伪装等规避手段，SOTA 模型仍大量漏判或误判。电商平台合规（logo 识别、品牌仿冒、商品类目合规）更缺乏专用评估框架。

**我们的设计：** Yuvion VL 构建三位一体解决方案：
1. 对抗感知数据飞轮（数据侧）
2. C2FT 对比微调（训练侧）
3. YVRE 三级评估框架（评估侧）

---

## 方法摘要 / Method Overview

### 架构概览
Yuvion VL 提供 8B 与 32B Dense 两种规格（各有 Instruct 和 Reasoning 版本），均在标准多模态底座上加入三阶段专用训练。

### 三大核心组件

**① 对抗感知数据生产流水线**
- 整合领域知识对齐 + CoT reasoning 标注 + 多级质量控制
- 数据飞轮：持续挖掘模型失败案例 → 有针对性增强训练数据
- 覆盖真实规避策略：小字/水印规避、logo 与符号变形、遮挡与隐藏、AI 生成对抗内容、上下文语义伪装

**② Confuse-then-Contrast Fine-Tuning（C2FT）**
- 在线挖掘模型自身"混淆对"（容易混淆的困难样例）
- 对多张图像进行联合对比监督
- 显著提升对对抗样本的细粒度安全判别能力

**③ Yuvion VL RiskEval（YVRE）**
三级渐进对抗评估框架：
- Level 1：开源通用 benchmark
- Level 2：开源安全 benchmark
- Level 3：自建能力与商业 benchmark（含电商治理）
  - Logo 识别
  - 品牌识别（含山寨检测）
  - 商品类目合规
  - 商品价格合规（异常价格识别）

---

## 与先前工作的差异 / Novelty vs Prior Work

| 维度 | 通用安全模型（如 Llama-Guard） | Yuvion VL |
|------|-------------------------------|-----------|
| 训练数据 | 静态安全数据集 | 动态对抗飞轮（持续更新） |
| 混淆处理 | 无针对性 | C2FT 在线挖掘混淆对 |
| 评估 | 通用安全 benchmark | 含电商治理的三级 YVRE |
| 实用场景 | 通用内容审核 | 达人违规检测 + 平台合规 |

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | Yuvion VL | 可比开源基线 | 更大闭源模型 |
|-------------|------|-----------|-------------|-------------|
| YVRE 综合 | Avg Score | SOTA | −9.9 pts | −6.7 pts |
| 多项安全任务 | Accuracy | 超越 GPT-5.4, Qwen3.5-Plus | — | — |
| 电商治理 benchmark | — | SOTA | — | — |

---

## 评分 / Score

| 维度 | 得分 | 最高 |
|------|------|------|
| Innovation | 25 | 30 |
| Experimental SOTA delta | 13 | 15 |
| Experimental quality / ablations | 12 | 15 |
| Efficiency | 6 | 10 |
| Generalization | 4 | 5 |
| Domain relevance (ecom + governance) | 22 | 25 |
| **Total** | **82** | **100** |

**评分理由：** 强领域相关性（自建电商治理 benchmark 直接覆盖达人违规场景）；C2FT 主动挖掘混淆对 + 数据飞轮闭环是训练范式创新；YVRE 将商业电商指标纳入评估是行业空白。效率扣分来自 8B/32B 模型的推理成本；泛化扣分来自数据飞轮的领域耦合。

---

## 代码复现 / Code Reproduction

位置：`code/YuvionVL/`（`2026-07-07/YuvionVL/`）

复现了 C2FT 的核心训练范式（混淆对挖掘 + 多图对比监督）以及 YVRE 评估框架的简化版本。
