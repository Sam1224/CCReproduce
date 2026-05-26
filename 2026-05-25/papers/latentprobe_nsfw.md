# Latent Space Probing for Adult Content Detection in Video Generative Models

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Latent Space Probing for Adult Content Detection in Video Generative Models |
| **arXiv** | [2605.00874](https://arxiv.org/abs/2605.00874) |
| **Authors** | (未完全公开，第一作者未明) |
| **Affiliations** | (不详) |
| **Date** | 2026-05-01 |
| **Bucket** | STRONG |
| **Total** | **72 / 100** |

---

## 故事弧 / Story Arc

> **问题:** AI视频生成系统（如CogVideoX）的内容安全检测现有两类方案均存在缺陷：(1) 提示词过滤：盲于生成过程中形成的内部表示；(2) 像素空间事后检测：计算开销高，无法实时介入生成流程。
>
> **方案:** 潜空间探测 (Latent Space Probing)——在视频扩散模型推理阶段，拦截去噪U-Net产生的潜在空间特征表示，附加轻量级分类器进行实时成人内容检测。生成过程本身成为检测介入点，无需等待最终像素输出。
>
> **差异:** 首次将探测分类器直接附加于视频扩散模型的内部去噪表示，而非作用于提示词或最终帧；实时性远优于像素空间方法；构建了规模最大的视频内容安全二分类数据集之一（11,039个视频片段）。

---

## 方法概述 / Method Summary

**框架概述:**

```
Text Prompt → CogVideoX Denoising U-Net
                  ↓ (at timestep t)
           Denoised Latent z_t  ← Probe Classifier attached
                  ↓
          Lightweight Classifier
          (CNN probe / MLP probe)
                  ↓
          Binary Label: Violating / Non-violating
```

**关键设计:**

1. **探测时机选择 (Probe Timestep):**
   - 在去噪过程的特定时间步拦截潜在特征（非所有步骤）
   - 早期步骤捕获语义内容，晚期步骤捕获视觉细节
   - 实验确定最优探测层和时间步

2. **轻量级探测分类器架构:**
   - 方案A: 轻量CNN探测器（卷积→池化→全连接）
   - 方案B: 轻量MLP探测器（展平→线性层）
   - 两种架构均设计为最小额外推理开销

3. **数据集构建:**
   - **违规:** 5,086个10秒视频片段（来源：成人网站）
   - **非违规:** 5,953个10秒视频片段（来源：YouTube）
   - 总计11,039个视频片段的大规模二分类数据集

**损失函数:**

$$\mathcal{L} = \mathcal{L}_{BCE}(\hat{y}, y) + \lambda \mathcal{L}_{reg}$$

---

## 创新性分析 / Innovation

1. **生成内部探测范式新颖:** 在扩散过程内部介入而非事后检测，开创视频AIGC安全检测新范式
2. **实时性突破:** 生成时同步检测，无需额外渲染开销
3. **数据贡献:** 11k+视频片段数据集是该细分领域较大规模的基准
4. **可泛化性局限:** 目前特定于CogVideoX，跨模型泛化需进一步验证

---

## 关键指标 / Key Metrics

| Dataset | Metric | Latent Probe | Pixel-space baseline |
|---------|--------|-------------|---------------------|
| 11,039 video clips | AUC | 高于像素空间方法 | 事后检测 |
| 11,039 video clips | ACC | 实时检测可达 | 需全渲染 |
| Latency | ms/clip | 显著更低 | 全渲染后检测 |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | 潜空间探测用于AIGC内容安全是新颖方向；但当前局限CogVideoX |
| Experimental SOTA delta | 10 | 15 | 大规模数据集上优于像素空间方法；详细数值有待查证 |
| Experimental quality / ablations | 10 | 15 | 数据集构建详细；消融时间步选择 |
| Efficiency | 8 | 10 | 实时检测是核心优势，明显优于事后方法 |
| Generalization | 3 | 5 | 当前针对CogVideoX，跨模型泛化待验证 |
| Domain relevance (ecom + governance) | 21 | 25 | AIGC视频成人内容检测是电商/短视频平台核心治理需求 |
| **Total** | **72** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **AIGC违规内容治理:** 电商直播、短视频平台需要高效检测AI生成违禁内容
- **实时内容审核:** 直播场景对毫秒级实时检测有强烈需求
- **产品图/视频合规:** AI生成商品图/达人内容视频的安全审核
- **数据集价值:** 11k视频安全数据集对平台内容治理研究有重要价值
