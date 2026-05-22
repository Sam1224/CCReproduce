# PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media |
| 作者 | Zoher Kachwala, Bao Tran Truong, Rasika Muralidharan, Haewoon Kwak, Jisun An, Filippo Menczer |
| 机构 | Observatory on Social Media (OSoMe), Indiana University, USA; Center Synergy of Systems, TUD Dresden University of Technology, Germany |
| arXiv | https://arxiv.org/abs/2605.17187 |
| 提交日期 | 2026-05-16 (v1); likely in May 21 arXiv listing |
| 领域标签 | 内容审核 · 多元化社区 · 多语言多模态基准 · VLM 评测 · 规则推理 |
| 桶类别 | STRONG |
| **总分 / Total** | **65 / 100** |

---

## 方法概述 (中文)

现代社交媒体平台正在向多元化社区治理演进——不同社区（如 Reddit subreddits）可以定义并执行各自的社区规范。现有内容审核基准大多基于通用平台级政策（如 Hate Speech），无法反映多元化社区语境下的规则异质性、文化多样性和多模态内容特性。

**PluRule** 构建了首个大规模多元化社区内容审核基准，核心特性：

1. **规模与多样性**: 覆盖 1,989 个 Reddit 社区、2,885 条规则、9 种语言，包含 13,371 条经人工验证的规则违规样本。
2. **任务定义**: 将审核形式化为多选题——给定一条评论及其上下文，识别违反了哪条（若有）具体规则。这与真实审核员工作方式一致。
3. **多模态多语言**: 涵盖图文混合内容，支持非英语社区（西班牙语、德语、日语等 9 种语言）。
4. **评测结果**: 即使是最先进的视觉语言模型（如 GPT-5.2 with high reasoning），在基准上的表现也仅略好于随机基线；通用规则（文明用语、自我推广）比长尾特定规则更易检测。

---

## 故事线 (Story Arc)

> **现状不足：** 现有内容审核基准（ToxiGen、HatEval 等）基于平台级静态有害类别，无法评测 AI 在社区规范多元化、规则动态、多语言场景下的审核推理能力。
>
> **我们的解法：** PluRule 通过覆盖 2K 社区/3K 规则/9 语言的大规模多元化基准，系统揭示当前 SOTA VLM 在"规则条件化推理"上的深层缺陷。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 首个聚焦多元化社区审核的多模态多语言大规模基准；将审核形式化为多选题式规则匹配 |
| vs. 先前工作 | 先前基准（SafetyBench、SALAD 等）专注通用有害内容；无法评测跨社区规则差异 |
| 可行性 | 基于真实 Reddit 数据和社区规则构建，有人工验证 |
| 局限 | 仅评测，无提升方案；Reddit 数据可能不完全代表中文/东亚平台场景 |

---

## 关键指标

| 数据集 | 模型 | 得分 | 随机基线 |
|--------|------|------|---------|
| PluRule benchmark | GPT-5.2 (high reasoning) | 略好于随机 | ~1/N rules |
| PluRule benchmark | 其他 SOTA VLM | ≈ random | ~1/N rules |
| 通用规则子集 (civility, self-promo) | GPT-5.2 | 明显高于平均 | — |
| 长尾社区规则子集 | 所有模型 | 接近随机 | — |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 18 | 30 | 多元化社区审核基准是新方向；但以数据收集为主，无新模型 |
| 实验 SOTA Delta | 8 | 15 | 揭示 SOTA 失败（diagnostic），非提升型贡献 |
| 实验质量/消融 | 13 | 15 | 13K 样本、9 语言、2K 社区；人工验证；多模型系统评测 |
| 效率 | 4 | 10 | 仅评测框架，无模型 |
| 泛化性 | 5 | 5 | 9 语言、2K 社区、多模态覆盖 |
| 领域相关性 | 17 | 25 | 内容治理高度相关；但侧重社交媒体通用规则而非电商特定场景 |
| **总分 / Total** | **65** | **100** | — |

---

## 代码与数据

- arXiv: https://arxiv.org/abs/2605.17187
- 数据集：基于 Reddit 公开数据构建，论文中提及计划开放
- 无代码复现（评测基准，分数 < 80）
