# Sparkle: Realizing Lively Instruction-Guided Video Background Replacement via Decoupled Guidance

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Sparkle: Realizing Lively Instruction-Guided Video Background Replacement via Decoupled Guidance |
| 作者 | Ziyun Zeng, Yiqi Lin, Guoqiang Liang, Mike Zheng Shou |
| 机构 | Show Lab, National University of Singapore |
| arXiv | https://arxiv.org/abs/2605.06535 |
| 提交日期 | 2026-05-07 |
| 领域标签 | 视频生成 · 内容编辑 · 达人内容 · 指令引导 · AIGV |
| 桶类别 | STRONG |
| 综合评分 | **82 / 100** |

---

## 方法概述 (中文)

短视频/直播达人在做内容创作时，背景替换是一个核心剪辑能力：要把人物或商品从原始场景中抠出，并替换到一个"看起来真实、随时间一致"的新背景里。现有方法把背景替换降级成"局部编辑或风格迁移"，结果背景不是太静态就是和前景脱钩。

**Sparkle** 把背景替换正式定义为"在保持前景-背景交互合理的前提下，合成全新的、时序一致的场景"。核心技术是**Decoupled Guidance**：分别为前景与背景生成引导信号，再用质量过滤管线进行筛选，使训练数据本身就携带"前景动作 ↔ 背景上下文"的对齐信号。

数据侧：构建 Sparkle 数据集（~140K 视频对）和 Sparkle-Bench 评测基准，覆盖五个主题。模型在 Sparkle-Bench 和 OpenVE-Bench 上同时超越现有方法。

---

## 故事线 (Story Arc)

> **现状不足：** 视频背景替换被现有工作降级处理，前景动作与背景物理交互失真，背景看上去"贴上去"而非"在那里"。
>
> **我们的解法：** Decoupled Guidance 把前景引导和背景引导分开生成 + 质量过滤，外加 Sparkle 数据集 / Sparkle-Bench 评测，实现指令引导的活泼背景替换。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | Decoupled Guidance 思路 + 配套大规模数据集 + 评测基准三件套 |
| vs. 先前工作 | OpenVE-Bench 偏局部编辑/风格迁移；Sparkle 把背景替换提升到"场景级合成"，且强调时序一致 |
| 可行性 | Show Lab 开源（项目页 https://showlab.github.io/Sparkle/）|

---

## 关键指标

| 数据集 | 指标 | 备注 |
|--------|------|------|
| Sparkle-Bench | 视频质量/时序一致 | 论文方法 SOTA |
| OpenVE-Bench | 现有评测 | Sparkle 超越现有方法 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 24 | 30 | Decoupled Guidance + 数据集/基准三件套 |
| 实验 SOTA delta | 13 | 15 | 双 benchmark SOTA |
| 实验质量/消融 | 13 | 15 | 140K 视频 + 自建评测 |
| 效率 | 6 | 10 | 标准视频扩散开销 |
| 泛化性 | 4 | 5 | 五主题；视频域内可迁移 |
| 领域相关性 | 22 | 25 | 直接服务达人短视频/电商视频内容生产 |
| **Total** | **82** | **100** | ✅ ≥80，进入复现 |

---

## 代码与数据

- 项目页：https://showlab.github.io/Sparkle/
- 数据集 Sparkle（140K 视频对）+ Sparkle-Bench 评测基准开源
- **本仓库提供完整 PyTorch 复现：** `code/Sparkle/`
