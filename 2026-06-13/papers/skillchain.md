# SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants |
| **ArXiv ID** | [2606.12984](https://arxiv.org/abs/2606.12984) |
| **Authors** | Yimin Hu, Mengtao Xu, Hao Guo, Yuheng Song, Xiaoyong Zhu, Bo Zheng |
| **Affiliation** | Alibaba Group |
| **Submitted** | 2026-06-11 |
| **Source** | HuggingFace June 13 daily listing |
| **Bucket** | STRONG — 电商图片助手、Agent Skill、可运营、多意图 |

---

## 方法概述 / Method Overview

**故事弧线：** 电商图片助手面对"同一张商品图触发多种用户意图"的长尾分布，容易出现格式混乱、工具调用错误、路由漂移等问题，且缺乏自动诊断修复的闭环机制。→ SkillChain 将每个意图的行为约束显式化为声明式 Skill 规范，并通过三阶段闭环自动演进 Skill 以适应流量变化与新意图。

**三阶段 Pipeline：**
1. **Stage 1 — Skill Creation**：从任务说明与轨迹中生成初始 Skill（工具调用/卡片结构/领域知识），经人工反思 Gate 保证初始质量；
2. **Stage 2 — Route Optimizer**：挖掘线上路由失败案例，对 Skill Description 做 update/merge/discard 操作，修复 intent→Skill 映射漂移；
3. **Stage 3 — Body Refinement**：双路径评估（规则结构路径 + LLM-Judge 语义路径）+ 聚合归因，将线上失败转化为可执行的 Skill Body 修复指令。

**关键创新：**
- "声明式 Skill"将约束显式化（vs 隐式学习），使结构化输出与工具调用可控；
- Route Optimizer 专门治理 intent routing drift（工业常见但学术鲜有）；
- 双路径评估 + 聚合归因实现从线上失败到修复指令的自动转化；
- 阶段解耦（Stage Gate）保证 Skill 质量单调不降。

---

## 关键指标 / Key Metrics

**离线（1000 条生产 query）：**

| 模块 | LLM-Judge Avg | Routing F1 |
|------|--------------|------------|
| Stage2 only | 63.4 | 70.5 |
| **Full SkillChain** | **72.2** | **73.5** |

细项：TCR 61.3 / CCC 72.2 / CQ 82.3 / CA 62.7（Full SkillChain）

**线上 A/B（Stage3 vs Stage2，1 周）：**

| 指标 | 变化 |
|------|------|
| Interactive UV | **+1.92pp** |
| Full-read Rate | **+4.98pp** |
| Avg Dwell Time | **+2.85s** |
| 7-day Return Rate | **+1.15pp** |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 22 | 30 | 声明式 Skill + routing drift 治理 + 双路径归因，工程创新突出 |
| 实验指标 | 12 | 15 | 离线 LLM-Judge + 线上 UV/留存双验证 |
| 实验质量 | 11 | 15 | 评测与数据高度生产定制，学术可复现性较低 |
| 方法效率 | 6 | 10 | Skill 演进是异步迭代，推理成本标准 |
| 方法泛化性 | 3 | 5 | 高度依赖 Alibaba 电商 Skill 生态 |
| 领域相关性 | 24 | 25 | 电商图片助手直接相关，含线上 A/B 验证 |
| **Total** | **78** | **100** | |

---

## Story Arc

> **现状不足：** 图片电商助手 Skill 定义分散、路由易漂移、上线后缺乏自动修复回路，导致意图覆盖和格式质量持续退化。  
> **解法：** Stage1 显式化 Skill → Stage2 路由漂移治理 → Stage3 双路径归因修复 → 线上 A/B 验证参与度和留存全面提升。
