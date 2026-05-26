# VINA: Video as Natural Augmentation — Towards Unified AI-Generated Image and Video Detection

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection |
| **arXiv** | [2605.21977](https://arxiv.org/abs/2605.21977) |
| **Authors** | Zhengcen Li, Chenyang Jiang, Liangxiu Su, Tong Shao, Shiyang Zhou, Ming Tao, Jingyong Su |
| **Affiliations** | Harbin Institute of Technology (Shenzhen), Pengcheng Laboratory, Shenzhen Loop Area Institute |
| **Date** | 2026-05-21 |
| **Bucket** | STRONG |
| **Total** | **83 / 100** |
| **Code** | `code/VINA/` (reproduction in this repo) |

---

## 故事弧 / Story Arc

> **现有问题 (Problem):** 当前最优的AIGC图像检测器在应用于视频帧时性能大幅下降——跨模态差距来源于视频处理流水线（色彩转换、编解码压缩、缩放、模糊）以及现代视频生成模型独有的频率指纹，使图像检测器"失明"于视频伪造。
>
> **设计方案 (Solution):** VINA (VIdeo as Natural Augmentation) 将视频帧作为物理基础的天然数据增强来联合训练检测器；通过**跨模态监督对比目标**将图像与视频帧的表示对齐，在统一特征空间中同时处理两种模态。无需复杂的手工增强或数据集特定调优。
>
> **现有工作差异 (Difference vs Prior Art):** 现有方法将图像检测和视频检测视为独立任务，分别设计流水线。VINA首次将视频作为自然增强源，以一个统一模型跨模态检测；是首个在14个图像、视频及野外基准上同时取得SOTA的方法。

---

## 方法概述 / Method Summary

VINA框架由三个核心组件构成：

1. **视频帧采样 (Video Frame Sampling):** 从AI生成/真实视频中均匀采样帧，作为图像检测器的训练增强数据。视频处理引入的自然变换（编解码、色彩空间变换等）充当物理上合理的数据增强。

2. **跨模态对比学习 (Cross-Modal Contrastive Objective):** 设计监督对比损失，将同一来源（真实/AI生成）的图像帧与视频帧对齐到统一嵌入空间，消除跨模态差距。
   - 正例对：同一生成器产生的图像帧与视频帧
   - 负例对：真实图像/帧 vs. AI生成图像/帧

3. **统一检测推理 (Unified Inference):** 单一模型不区分输入是图像还是视频帧，统一前向推理完成检测，无需针对不同来源进行特定校准。

**公式 (Contrastive Loss):**

$$\mathcal{L}_{VINA} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{SCL}$$

其中 $\mathcal{L}_{SCL}$ 是跨模态监督对比损失：

$$\mathcal{L}_{SCL} = -\frac{1}{N} \sum_{i} \log \frac{\exp(z_i \cdot z_i^+ / \tau)}{\sum_j \exp(z_i \cdot z_j / \tau)}$$

---

## 创新性分析 / Innovation

1. **核心洞察新颖:** 将视频理解为"自然增强"而非独立任务，颠覆了领域惯例——无需额外标注或模型修改
2. **统一检测无需双路径:** 一个模型同时检测AI生成图像和视频，避免维护两套独立系统
3. **物理合理性:** 视频编解码引入的变换比人工增强更接近真实世界分布偏移
4. **可行性评估:** 方法简洁，实验覆盖14个基准，结果可重复；适合工业平台部署

---

## 关键指标 / Key Metrics

| Dataset | Metric | VINA | Prior SOTA |
|---------|--------|------|-----------|
| 14 image/video benchmarks (avg) | AUC | SOTA | — |
| AI-generated image benchmarks | AUC | +双向提升 | Isolated image detector |
| AI-generated video benchmarks | ACC | SOTA | Isolated video detector |
| In-the-wild benchmarks | AUC | SOTA | — |

> *论文报告在几乎所有14个评估设置中均取得SOTA，且无需数据集特定调优。*

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 24 | 30 | 跨模态视频增强思路新颖；统一检测范式是领域首创 |
| Experimental SOTA delta | 13 | 15 | 14个基准上全面SOTA，覆盖广泛 |
| Experimental quality / ablations | 12 | 15 | 扩展实验充分；消融验证各组件贡献 |
| Efficiency | 7 | 10 | 单模型推理无额外开销；训练略增加视频数据处理 |
| Generalization | 5 | 5 | 14个不同基准验证泛化性 |
| Domain relevance (ecom + governance) | 22 | 25 | AIGC检测直接服务内容治理；视频平台/直播场景高度相关 |
| **Total** | **83** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **内容治理:** 电商平台、短视频平台（TikTok/快手/抖音）大量处理UGC视频，AI生成视频检测是违规内容检测的核心需求
- **直播监控:** 直播场景中实时检测AI换脸、AI生成背景等违规行为
- **达人内容合规:** KOL/达人内容的AIGC检测确保内容真实性合规
- **代码已复现:** `code/VINA/` 提供完整PyTorch实现
