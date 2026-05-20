# E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs |
| **Authors** | Xianjie Liu, Yiman Hu, Liang Wu, Ping Hu, Yixiong Zou, Jian Xu, Bo Zheng |
| **Affiliation** | Alibaba Group (Taobao) |
| **arXiv** | https://arxiv.org/abs/2602.08355 |
| **Submitted** | February 2026 |
| **Domain** | E-commerce · Short Video Understanding · MLLM · Benchmark · Commercial Intent Reasoning |
| **Code** | `code/EVAds/` |

---

## 方法概述 / Method Summary

电商短视频（如淘宝直播切片、带货视频）具有**极高的多模态信息密度**，现有视频理解基准（ActivityNet、EgoSchema 等）主要聚焦通用任务，无法评估模型对商业意图（commercial intent）的理解能力。

本文贡献：
1. **E-VAds 基准：** 来自淘宝的 3,961 条高质量电商短视频 + 19,785 个开放式 Q&A 对，涵盖多品类，覆盖商品识别、营销意图、用户价值理解等维度
2. **多模态信息密度框架：** 提出量化评估框架，证明电商视频在视觉/音频/文本模态的信息密度**显著高于**主流数据集
3. **E-VAds-R1 模型：** 基于 E-VAds 数据集，用多智能体标注系统生成高质量训练样本，结合强化学习（RL）微调，仅用数百条样本即可在商业意图推理上获得 109.2% 性能提升

---

## 故事弧线 / Story Arc

> 现有 MLLM 评测基准忽略了电商短视频的商业意图理解需求 → 构建 E-VAds 首个电商短视频基准并量化信息密度差距 → E-VAds-R1 通过 RL 微调实现少样本下商业意图推理的跨越式提升

---

## 创新分析 / Innovation Analysis

**与前人工作的差异：**
- 据作者所称，E-VAds 是**首个专为电商短视频设计的 MLLM 评测基准**
- 信息密度量化框架提供了客观标准来区分"为什么电商视频更难理解"
- 多智能体标注系统可规模化生成高质量电商视频 Q&A，解决了标注成本瓶颈
- E-VAds-R1 的强化学习方案仅需少量样本即可大幅提升商业意图推理，效率极高
- 代码+数据集开源（Taobao 平台），有助于推动领域研究

**可行性：** 淘宝真实数据，多智能体标注流程可复现，RL 微调有详细实验。

---

## 关键指标 / Key Metrics

| Dataset/Metric | Method | Value vs Baseline |
|----------------|--------|-------------------|
| E-VAds (commercial intent reasoning) | E-VAds-R1 (RL fine-tuned) | **+109.2%** vs base MLLM |
| E-VAds (video understanding) | State-of-the-art MLLMs | substantially lower than human |
| Information density | E-commerce video vs. mainstream datasets | significantly higher (all modalities) |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | 首个电商短视频基准，多智能体标注，密度量化框架 |
| Experimental SOTA delta | 12 | 15 | 109.2% 商业意图提升，少样本强化学习 |
| Experimental quality | 12 | 15 | 多智能体标注+RL，但外部验证尚待同行评审 |
| Efficiency | 8 | 10 | 百样本级 RL 微调即可大幅提升 |
| Generalization | 4 | 5 | 跨品类，多 MLLM 评测 |
| Domain relevance | 24 | 25 | 核心命中：电商短视频理解 + 商业意图 + 标注 |
| **Total** | **84** | **100** | |

---

## 代码复现说明

完整 PyTorch 复现位于 `code/EVAds/`

核心组件：
- `data/`: 合成电商短视频 Q&A 数据集（toy 接口对齐真实 E-VAds）
- `model/`: MLLM 微调封装（支持 LLaVA/InternVL 接口）
- `annotation/`: 多智能体自动标注 pipeline
- `train_rl.py`: 强化学习微调脚本（基于 PPO/GRPO）
- `eval.py`: E-VAds 评测脚本
