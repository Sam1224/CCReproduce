# Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance |
| **Authors** | Jiancheng Zhang, Meiqing Li, Qi Zhang, Yinglun Zhu |
| **Affiliation** | University of California Riverside; Carnegie Mellon University; Worcester Polytechnic Institute |
| **ArXiv** | [2606.07630](https://arxiv.org/abs/2606.07630) |
| **Submitted** | May 30, 2026 |
| **Domain Tags** | `data-labeling` `active-learning` `data-quality` `foundation-models` `class-imbalance` |
| **Total** | **64 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 真实世界数据集（图像和文本）通常同时存在类别分布不均衡和噪声标注两大挑战，联合降低模型性能，尤其对少数类的损害更大。

**解决方案：** 提出利用基础模型先验（Foundation Model Priors）的主动学习框架，通过基础模型与小模型的不均衡感知协同决策，同时应对噪声标签和类别不均衡问题。

---

## 方法概述 / Method

**核心框架：**

1. **Foundation Model Prior 注入：** 利用 CLIP 等基础模型的零样本/少样本能力提供样本难度估计和类别先验信息，无需额外标注数据。

2. **Imbalance-Aware Co-Decision：** 基础模型（大）与任务模型（小）协同决策：
   - 基础模型提供可靠的类别分布估计
   - 任务模型提供任务特定的不确定性
   - 两者结合选择最具信息量且能缓解类别不均衡的样本

3. **Noise-Robust Sample Selection：** 引入噪声检测机制，在主动学习循环中过滤噪声标注样本，避免模型在错误标签上过拟合。

**适用场景：** 图像和文本双域，适用于大规模数据标注场景（如电商商品属性标注、内容审核标注）。

---

## 创新性分析 / Innovation

**首次系统性研究同时应对噪声标签和类别不均衡的主动学习框架**，利用基础模型先验降低标注成本的同时提升数据质量。

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| Imbalanced image datasets | Annotation savings | Substantial reduction |
| Text classification | Minority class accuracy | Significant improvement |
| Both domains | Novel first study | Noise + imbalance combined |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 18 | 30 | First systematic study on noise + imbalance in active learning with FM priors |
| Experimental SOTA delta | 10 | 15 | Substantial annotation savings demonstrated |
| Experimental quality / ablations | 11 | 15 | Extensive experiments across image and text domains |
| Efficiency | 7 | 10 | Annotation efficiency is core contribution |
| Generalization | 4 | 5 | Both image and text domains |
| Domain relevance | 14 | 25 | Data labeling quality for large-scale annotation — moderately relevant |
| **Total** | **64** | **100** | |

---

## 中文摘要

本文研究同时存在类别不均衡和噪声标注的主动学习问题，这是大规模数据标注场景（如电商商品标注、内容审核标注）的真实挑战。提出基于基础模型先验（Foundation Model Priors）的主动学习框架：利用 CLIP 等基础模型提供类别分布估计和样本难度先验，与任务模型协同做出不均衡感知的标注决策，同时内嵌噪声检测机制过滤错误标签。这是首个在图像和文本双域系统性研究噪声+不均衡联合挑战的主动学习工作，可大幅降低高质量标注的成本。
