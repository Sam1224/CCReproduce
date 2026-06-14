# EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments |
| **ArXiv ID** | [2606.13681](https://arxiv.org/abs/2606.13681) |
| **Authors** | Jundong Xu, Qingchuan Li, Jiaying Wu, Yihuai Lan, Shuyue Stella Li, Huichi Zhou, Bowen Jiang, Lei Wang, Jun Wang, Anh Tuan Luu, Caiming Xiong, Hae Won Park, Bryan Hooi, Zhiyuan Hu |
| **Affiliation** | NUS, SMU, UW, UCL, UPenn, NTU, Recursive, MIT |
| **Submitted** | 2026-06-13 |
| **Source** | HuggingFace June 13 daily listing (GitHub mirror) |
| **Bucket** | WEAK — Agent 记忆、动态环境鲁棒性 |

---

## 方法概述 / Method Overview

**故事弧线：** 多数 LLM Agent 评测假设静态环境快照，而真实部署中工具 API/工作流/代码库/用户偏好持续迭代演化，导致 Agent 记忆过时、知识状态坍缩。→ EvoArena 将环境变化建模为 release chain（版本序列），并提出 EvoMem：像 git 一样记录记忆的 patch 历史（更新前后内容、变更理由与证据），支持版本回溯与冲突解决。

**EvoArena Benchmark：**
- 三个演化领域：Terminal（系统命令变化）、Software（API/SDK 更新）、Social-preference（用户偏好漂移）；
- 任务被组织为 release chain（顺序更新版本）；
- Chain-level accuracy 度量 Agent 在连续演化任务中的累积成功率。

**EvoMem：**
- Append-only patch 结构：每次更新记录 (before, after, rationale, evidence)；
- 默认用最新状态，冲突时检索历史 patch；
- 与任意 LLM backbone 兼容。

---

## 关键指标 / Key Metrics

| 指标 | 基础 Agent | + EvoMem | 提升 |
|------|-----------|---------|------|
| EvoArena avg accuracy | ~39.6% | ~41.1% | +1.5% |
| Chain-level accuracy | base | base + | +3.7% |
| GAIA benchmark | base | base + | +6.1% |
| LoCoMo benchmark | base | base + | +4.8% |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 22 | 30 | Git-like patch memory 概念新颖；EvoArena 是首个版本演化 benchmark |
| 实验指标 | 9 | 15 | 绝对提升幅度中等（+1.5%/+3.7%），但在新 benchmark 上有意义 |
| 实验质量 | 13 | 15 | 三领域覆盖，多 backbone 对比，chain 级指标设计合理 |
| 方法效率 | 6 | 10 | 轻量 memory 结构，额外开销小 |
| 方法泛化性 | 4 | 5 | 三域通用；patch 结构适用于任何动态工具/流程 |
| 领域相关性 | 14 | 25 | 对内容治理 Agent 工具链持续迭代有借鉴，但非直接电商/内容场景 |
| **Total** | **68** | **100** | |

---

## Story Arc

> **现状不足：** 静态评测高估 Agent 可靠性；环境版本迭代会导致记忆失效、路由冲突和行为退化。  
> **解法：** EvoArena 版本链 benchmark 暴露问题 → EvoMem patch history 显式化演化记忆 → 跨 backbone 验证链级任务鲁棒性改善。

---

## 对内容治理的启发

电商达人治理系统中，政策规则、品类白名单、禁词库持续更新。EvoMem 的"patch history + 冲突解决"范式可以直接应用于：
- 内容审核 Agent 的政策知识库版本管理；
- 达人违规判定规则的增量更新与回溯；
- 工具链（API/审核模型）升级时的无缝记忆迁移。
