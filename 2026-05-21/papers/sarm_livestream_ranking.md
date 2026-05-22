# SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking |
| 作者 | Ruochen Yang, Yueyang Liu, Zijie Zhuang, Changxin Lao, Yuhui Zhang, Jiangxia Cao, Jia Xu, Xiang Chen, Haoke Xiao, Xiangyu Wu, Xiaoyou Zhou, Xiao Lv, Shuang Yang, Tingwen Liu, Zhaojie Liu, Han Li, Kun Gai |
| 机构 | 快手科技 (Kuaishou Technology) |
| arXiv | https://arxiv.org/abs/2602.09401 |
| 提交日期 | 2026-02-10 (v1) — *注：早于 May 21 窗口，作为参考* |
| 领域标签 | 直播推荐 · 语义锚点 · LLM · 端到端排序 · 工业部署 |
| 桶类别 | STRONG (reference only — outside May 21 window) |
| **总分 / Total** | **85 / 100** |

---

## 方法概述 (中文)

大规模直播推荐需要在严格实时约束下对非平稳多模态内容语义进行精确建模。现有工业方案有两类缺陷：
- **离散语义抽象**（聚类/类目标签）：牺牲描述精度；
- **密集多模态嵌入**（独立提取，弱对齐）：与排序优化目标脱节。

**SARM** 提出语义锚点（Semantic Anchor）机制：

1. **可学习语义锚点（Learnable Semantic Anchor Text Tokens）**: 将每个主播/直播间的语义表示设计为一组可学习文本 token，这些 token 与排序特征端到端联合优化——语义锚点随排序目标的反向传播自适应更新内容描述，而非依赖预定义分类。
2. **轻量级双 Token 门控设计（Dual-Token Gated Design）**: 专为直播场景设计的双 token 结构，分别捕获领域专属语义和通用语义，门控机制控制融合比例。
3. **非对称部署策略（Asymmetric Deployment）**: 在线训练（含 LLM）和在线服务（轻量 anchor 推理）解耦，保证线上低延迟服务同时维持 LLM 语义质量。

---

## 故事线 (Story Arc)

> **现状不足：** 直播内容语义高度动态（主播风格变化、商品切换、互动随机），传统基于聚类的离散标签牺牲精度，密集嵌入与排序目标弱对齐——两者均无法真正将内容语义注入排序优化。
>
> **我们的解法：** SARM 引入可学习语义锚点 token，直接参与端到端排序训练——让模型"以语言描述驱动排序"，并通过非对称部署保持生产级推理效率。已在快手 4 亿用户规模 A/B 测试验证。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 语义锚点 token 与排序特征端到端联合优化：首次让 LLM 文本描述直接影响排序 loss 梯度 |
| vs. 先前工作 | 先前方法（DLRM、PinnerFormer 等）将内容特征作为固定输入；SARM 使锚点与任务目标同步演化 |
| 可行性 | 全面部署在快手，4 亿日活用户，线上 A/B 测试显著正向 |
| 局限 | 系统依赖专有直播数据；非对称部署依赖 LLM 推理在训练侧的可扩展性 |

---

## 关键指标

| 评测 | 指标 | SARM | 生产基线 |
|------|------|------|---------|
| 快手线上 A/B Test | 用户在线时长 | ↑ 显著提升 | 生产系统 |
| 快手线上 A/B Test | 互动指标 (点赞/评论/关注) | ↑ 显著提升 | 生产系统 |
| 离线评测 | 多指标综合 | 一致改善 | 多个生产基线 |
| 部署规模 | 日活用户 | **4 亿+** | — |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 端到端语义锚点机制新颖；双 token 门控设计有工程创新 |
| 实验 SOTA Delta | 13 | 15 | 4 亿用户规模 A/B 测试，线上正向，工业可信度极高 |
| 实验质量/消融 | 14 | 15 | 大规模线上实验 + 离线评测；消融分析充分 |
| 效率 | 9 | 10 | 非对称部署保证实时性；已在生产环境验证 |
| 泛化性 | 5 | 5 | 4 亿用户规模，覆盖多场景 |
| 领域相关性 | 22 | 25 | 直播排序直接对应达人/主播推荐核心场景 |
| **总分 / Total** | **85** | **100** | — |

---

## 代码与数据

- arXiv: https://arxiv.org/abs/2602.09401
- 无公开代码（生产系统）
- *注：此论文提交于 2026-02-10，早于本次 May 21 检索窗口，作为领域参考列出。*
