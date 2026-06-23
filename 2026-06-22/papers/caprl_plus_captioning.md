# CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning |
| 作者 | Penghui Yang, Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Yibin Wang, Yujie Zhou, Jiazi Bu, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, Dahua Lin |
| 机构 | InternLM / Shanghai AI Lab / CUHK |
| arXiv | https://arxiv.org/abs/2606.09393 |
| GitHub | https://github.com/InternLM/CapRL |
| 提交日期 | 2026-06-08 |
| 领域标签 | 图像描述 · 视频描述 · 强化学习 · RLVR · 多模态 · LVLM |
| 桶类别 | WEAK |
| 综合评分 | **75 / 100** |

---

## 方法概述 (中文)

图像和视频描述（Captioning）是大型视觉语言模型（LVLM）的基础预训练任务，高质量描述数据对下游理解能力至关重要。现有方法主要依赖**监督微调（SFT）**，存在三大缺陷：
1. **标注昂贵且不可扩展**：需要人工标注精细描述；
2. **记忆效应**：模型记忆特定 ground-truth 答案，缺乏多样性；
3. **无法评估"描述质量"的绝对标准**：传统指标（BLEU/CIDEr）与人类偏好相关性弱。

**CapRL++** 将**可验证奖励强化学习（RLVR）**引入开放式 Captioning 任务：

**核心思路**：高质量描述应使**无视觉的语言模型**能通过阅读该描述来准确回答视觉问题。

1. **验证奖励机制**：LVLM 生成一段描述 → 将描述（非图像）送给独立的 vision-free LLM → LLM 基于描述回答多选视觉问题（MCQ）→ 回答准确率作为 RL 奖励信号。奖励完全可验证（MCQ 有标准答案），无需人工偏好标注。

2. **图像→视频统一扩展**：对视频任务增加时序格式奖励（时间戳标注奖励）和长度正则化（防止冗余描述），同一 RLVR 框架统一训练 Qwen3-VL 风格基础模型的图像+视频 Captioning。

3. **质量提升**：在超过 20 个图像和视频 benchmark 上改善描述密度和质量，基于 CapRL++ 训练的紧凑模型在 Prism 框架评估下可媲美 Qwen2.5-VL-72B 和 Qwen3-VL-235B 等超大模型。

---

## 故事线 (Story Arc)

> **现状不足：** SFT 做 Captioning 需要昂贵标注、容易过拟合，且描述多样性受限；传统 CIDEr/BLEU 分不代表真正的信息密度。
>
> **我们的解法：** CapRL++ 用"描述能帮盲模型答对问题"作为可验证奖励，将 RLVR 从推理任务推广到开放式 Captioning，并统一图像/视频，在无需新标注的情况下大幅提升描述质量和密度。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 首次将 RLVR 应用于图像/视频 Dense Captioning；可验证奖励信号无需人工偏好标注 |
| 验证奖励设计 | MCQ 准确率作为奖励，与描述信息完整性直接相关，理论上更接近真实质量 |
| 图像-视频统一 | 时序奖励+长度正则化，单一 RL 框架统一两类 Captioning 任务 |
| vs. 先前工作 | CapRL（ICLR 2026 前身）专注图像；CapRL++ 扩展至视频并引入统一框架 |
| 局限 | MCQ 覆盖的"视觉问题空间"可能不能全面代表描述质量；视频时序密度评估尚在探索 |

---

## 关键指标

| 实验 | 数据集/场景 | 指标 | CapRL++ | 对比 |
|------|------------|------|---------|------|
| 图像描述质量 | Prism Framework | 紧凑模型 vs Qwen2.5-VL-72B | 可媲美 | Qwen2.5-VL-72B |
| 图像描述质量 | Prism Framework | 紧凑模型 vs Qwen3-VL-235B-A22B | 可媲美 | 235B 超大模型 |
| 综合评测 | 20+ 图像+视频 benchmark | 空间理解、时序理解 | 全面提升 | SFT baseline |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | RLVR 应用于开放式 Captioning 是真正的方法论突破；可验证奖励设计优雅 |
| Experimental SOTA delta | 12 | 15 | 紧凑模型媲美超大模型；20+ benchmark 全面验证 |
| Experimental quality | 13 | 15 | 图像+视频统一评测，消融实验充分，GitHub 代码开源 |
| Efficiency | 7 | 10 | 无需标注数据扩展，但 RL 训练计算成本高 |
| Generalization | 5 | 5 | 框架可推广至任意 LVLM 的 Captioning 训练 |
| Domain relevance | 14 | 25 | Captioning 是电商内容理解的基础，可迁移到商品/达人视频描述；但非直接电商应用 |
| **Total** | **75** | **100** | |

---

## 相关链接

- arXiv: https://arxiv.org/abs/2606.09393
- GitHub: https://github.com/InternLM/CapRL
- 前身论文 CapRL: https://arxiv.org/abs/2509.22647 (ICLR 2026)
