# DisAAD: Estimating Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation |
| 作者 | Huizi Cui, Huan Ma, Qilin Wang, Yuhang Gao, Changqing Zhang |
| 机构 | 天津大学 (Tianjin University) |
| arXiv | https://arxiv.org/abs/2605.05777 |
| 提交日期 | 2026-05-07 (v1) |
| 领域标签 | LLM 不确定性 · 黑盒蒸馏 · 对抗训练 · 幻觉检测 · 代理模型 |
| 桶类别 | WEAK |
| 综合评分 | **57 / 100** |

---

## 方法概述 (中文)

大语言模型幻觉问题是制约商业 API（GPT、Claude 等黑盒 LLM）实际部署的核心瓶颈。现有不确定性量化方法要么依赖多次昂贵采样（token 概率方法），要么需要访问模型内部参数（白盒方法），在黑盒场景下均无法实用化。

**DisAAD（Distribution-Aligned Adversarial Distillation）** 提出生成-判别对抗架构：

1. **代理生成器**：一个轻量级代理模型（proxy model）学习黑盒 LLM 的高质量输出分布区域
2. **对抗判别器**：判别器引导代理模型对齐黑盒 LLM 的输出分布，通过对抗训练确保代理模型能捕捉黑盒 LLM 的"知道与不知道"边界
3. **分布对齐目标**：通过最小化代理模型输出分布与黑盒 LLM 真实输出分布之间的 Jensen-Shannon 散度，实现无需白盒访问的不确定性估计

训练完成后，轻量代理模型可实时估计黑盒 LLM 的回答置信度，无需额外 API 调用。

---

## 故事线 (Story Arc)

> **现状不足：** 商业黑盒 LLM（GPT-4、Claude）仅提供文本输出，无法获取内部置信度；多次采样成本极高；幻觉检测无法实时化。
>
> **我们的解法（天津大学）：** DisAAD 训练对抗蒸馏代理模型，学习黑盒 LLM 的分布"高质量区域"，实现低成本实时不确定性估计，让用户知道黑盒 LLM "知不知道"。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 生成-判别对抗架构实现黑盒 LLM 分布对齐蒸馏；无需白盒访问 |
| vs. 先前工作 | P(True) 等方法需要多次采样；Conformal Prediction 需内部概率；DisAAD 仅需文本输出 |
| 可行性 | 在多个 QA 基准验证；轻量代理模型可实际部署 |
| 局限 | 代理模型的训练数据分布影响蒸馏质量；跨任务泛化性待验证 |

---

## 关键指标

| 数据集 | 指标 | DisAAD | 基线 |
|--------|------|--------|------|
| 多个问答基准（均值） | 不确定性估计 AUROC | 高于多采样基线 | 多次采样法 |
| 推理效率 | 额外 API 调用次数 | **0** | 多次采样（5–20 次） |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 18 | 30 | 对抗蒸馏思路有意思，但 GAN-style 蒸馏并非全新 |
| 实验 SOTA Delta | 9 | 15 | 与多采样基线竞争，效率优势明显 |
| 实验质量/消融 | 9 | 15 | 基准测试合理，消融略显不足 |
| 效率 | 8 | 10 | 零额外 API 调用；代理模型轻量 |
| 泛化性 | 3 | 5 | 跨任务泛化有限 |
| 领域相关性 | 10 | 25 | LLM 不确定性对电商问答质量控制有参考价值，但间接 |
| **总分** | **57** | **100** | — |

---

## 代码与数据

- 天津大学研究项目（代码状态未确认）
