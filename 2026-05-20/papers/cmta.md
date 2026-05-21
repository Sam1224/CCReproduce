# CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection |
| **作者** | Hang Wang, Chao Shen, Chenhao Lin, Minghui Yang, Lei Zhang, Cong Wang |
| **机构** | Xi'an Jiaotong University; The Hong Kong Polytechnic University; City University of Hong Kong; Guangdong OPPO Mobile Communications |
| **arXiv ID** | [2605.00630](https://arxiv.org/abs/2605.00630) |
| **提交日期** | May 1, 2026 |
| **代码** | Not yet public |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** Existing AI-generated video (AIGV) detection methods focus on single-modal or spatiotemporal visual artifacts, missing rich cues in the visual-textual cross-modal space. CMTA identifies a novel fingerprint called the **cross-modal temporal artifact (CMTA)**: real videos exhibit natural, semantically-driven fluctuations in visual-text alignment over time (because scenes change meaning), whereas AI-generated videos display *unnaturally stable* semantic trajectories — they are "locked" to the generating text prompt throughout the clip.

The framework leverages **BLIP** to generate frame-level image captions, uses **CLIP** to extract visual-textual embeddings, and feeds temporal sequences of cross-modal alignment scores into a dual-branch architecture: (1) a coarse-grained temporal branch using GRU to model temporal fluctuation patterns, and (2) a fine-grained semantic branch to capture local alignment anomalies. Final detection is a binary classifier over concatenated branch representations.

**ZH:** CMTA 发现了一种新型 AI 生成视频指纹：真实视频中视觉-文本跨模态对齐度会随场景语义变化而自然波动，而 AI 生成视频受生成提示词"锁定"，呈现异常稳定的跨模态语义轨迹。方法用 BLIP 生成帧级字幕、CLIP 提取跨模态表征，通过粗粒度（GRU）和细粒度语义两分支检测时序异常，最终输出真实/生成二分类。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 现有 AIGV 检测器仅关注单模态（视觉帧）或时空伪影，忽视了生成式视频在视觉-语言跨模态一致性上的独特规律。
>
> **我们的解决方案:** 提出跨模态时序伪影（CMTA）这一新型指纹概念：通过测量视频片段内视觉-文本对齐随时间的稳定性，以低参数量实现对多种 AI 生成模型（Sora、Runway Gen-3、Kling 等）的泛化检测。

---

## 创新性分析 / Innovation Analysis

1. **新型指纹概念：** CMTA 是首次从跨模态时序对齐稳定性角度定义 AI 生成视频特征，不依赖特定生成模型的视觉伪影。
2. **高泛化性：** 不依赖单一生成模型的训练数据，在跨模型评估中保持良好性能。
3. **轻量化设计：** 基于 BLIP+CLIP 表征提取，仅需轻量 GRU 分类器，无需大型解码器。
4. **vs. 先前工作：** FakeSV、AltFreezing 等依赖视觉频域伪影，CMTA 利用完全不同的跨模态信号，有互补价值。

---

## 关键指标 / Key Metrics

| Dataset | Metric | CMTA | Best Prior |
|---------|--------|------|------------|
| GenVideo-Bench (cross-gen) | AUC | ~93.4% | ~88.7% |
| FakeShield-Video | Acc | ~89.1% | ~84.3% |
| In-distribution | AUC | ~96.2% | ~94.5% |

> *Values approximate from paper reporting; generalization benchmark most informative.*

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 23/30 | 跨模态时序伪影概念新颖，开辟新检测维度 |
| Experimental SOTA delta | 11/15 | 跨模型泛化提升明显（+4~5pp AUC） |
| Experimental quality | 12/15 | 多数据集、跨生成器评估 |
| Efficiency | 8/10 | 轻量 GRU 分类器，生产友好 |
| Generalization | 5/5 | 跨 Sora/Runway/Kling 等多种生成器 |
| Domain relevance | 18/25 | 直接对应电商平台 AI 生成视频内容检测需求 |
| **Total** | **77/100** | |
