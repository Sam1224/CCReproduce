# ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning |
| 作者 | Zuhao Yang, Kaichen Zhang, Sudong Wang, Keming Wu, Zhongyu Yang, Bo Li, Xiaojuan Qi, Shijian Lu, Xingxuan Li, Lidong Bing |
| 机构 | EvolvingLMMs-Lab 等 |
| arXiv | https://arxiv.org/abs/2605.20342 |
| GitHub | https://github.com/EvolvingLMMs-Lab/ParaVT |
| 提交日期 | 2026-05-19 (v1)；HuggingFace Daily 2026-05-26 |
| 领域标签 | 视频理解 · 多模态 Agent · 工具调用 · 强化学习 · 长视频推理 |
| 桶类别 | MODERATE |
| 综合评分 | **67 / 100** |

---

## 方法概述 (中文)

当前将大型多模态模型（LMM）训练为视频工具调用 Agent 的方案均采用**顺序工具调用**（每轮一次裁剪），导致三个核心缺陷：①单次错误裁剪无同伴校正，错误传播；②多轮工具调用污染上下文窗口；③推理成本随调用轮次线性增长。

**ParaVT** 提出首个端到端 RL 训练的**并行视频工具调用**多 Agent 框架：

1. **并行调度**：主 Agent 在单轮内同时发出多个时间窗口裁剪请求（如同时截取 t=0~30s、t=60~90s、t=120~150s 三段），而非逐轮顺序调用。
2. **子 Agent 权重共享处理**：多个权重共享的子 Agent 分别处理各裁剪段，聚合并行证据后由主 Agent 合成最终答案。
3. **工具先验悖论（Tool Prior Paradox）识别**：作者首次识别并命名这一训练障碍——预训练工具先验（pretrained tool priors）既帮助模型探索工具使用，又妨碍 RL 训练收敛，二者相互矛盾。
4. **PARA-GRPO 算法**：在标准 GRPO（Group Relative Policy Optimization）基础上，增加两个互补机制：① **可解析性锚定格式奖励（Parseability-Anchored format reward）**，仅在最易结构崩溃的 token 位置施加靶向格式奖励；② **帧预算随机化（Ratio-gated frame-budget randomization）**，对每个 prompt 随机化帧预算，创造"调用工具比不调用奖励更高"的训练信号。

---

## 故事线 (Story Arc)

> **现状不足：** 视频 Agent 顺序工具调用导致错误传播、上下文污染和线性推理成本，且 RL 训练时预训练工具先验引入训练悖论，难以收敛。
>
> **我们的解法（ParaVT）：** 多 Agent 并行调度 + PARA-GRPO 同时解决架构效率问题和训练稳定性问题，首次实现端到端 RL 训练的并行视频工具调用。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 首次识别"工具先验悖论"并提供配套 PARA-GRPO 解决方案；首个并行视频工具调用 RL 框架 |
| vs. 先前工作 | 现有方法（如 LLaVA-OneVision、InternVL Video）仅支持顺序工具调用；ParaVT 引入并行多 Agent 架构 |
| 可行性 | 开源代码（GitHub），理论分析充分；并行推理减少轮次已有实测支撑 |
| 局限 | 仅验证在视频工具（crop）场景；对其他工具类型泛化需要进一步探索 |

---

## 关键指标

| 数据集/基准 | 指标 | ParaVT | 顺序调用基线 |
|------------|------|--------|------------|
| 长视频 QA 基准 | 准确率 | 显著优于顺序调用 | 基线 |
| 推理轮次 | 工具调用轮数 | 1 轮（并行） | N 轮（顺序） |
| 错误容忍性 | 单裁剪错误影响 | 降低（多子 Agent 互校） | 高（单点传播） |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 工具先验悖论命名+解决方案设计新颖；并行多 Agent RL 首创 |
| 实验 SOTA Delta | 11 | 15 | 首创框架对比顺序基线有效，但尚缺与通用 SOTA 对比 |
| 实验质量/消融 | 11 | 15 | 有开源代码，含消融验证 PARA-GRPO 各组件贡献 |
| 效率 | 7 | 10 | 并行调度显著减少推理轮次，上下文污染降低 |
| 泛化性 | 3 | 5 | 当前局限于视频裁剪工具；框架设计可扩展 |
| 领域相关性 | 13 | 25 | 视频 Agent 框架可用于直播电商内容审核、达人视频理解，但非直接电商应用 |
| **Total** | **67** | **100** | — |

---

## 代码与数据

- 代码仓库：https://github.com/EvolvingLMMs-Lab/ParaVT（已开源）
- 评分未达 80，使用官方代码
