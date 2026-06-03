# CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning |
| **Authors** | Jun-Tao Tang, Zhen-Hao Xie, Yu-Cheng Shi, Da-Wei Zhou |
| **Affiliations** | (Not fully disclosed; from abstract pages) |
| **Link** | https://arxiv.org/abs/2606.02502 |
| **Submission Date** | 2026-06-01 (appears in 2026-06-02 listing) |
| **Domain Bucket** | **WEAK** — multimodal continual instruction tuning; applicable to adapting MLLM to new e-commerce categories |
| **Total** | **65 / 100** |

---

## 方法概述 / Method Overview

### 问题背景 (Problem)
多模态大语言模型（MLLM）在实际部署中需要**持续拓展能力**（Multimodal Continual Instruction Tuning, MCIT）：随着新任务、新商品类目、新审核标准不断涌现，模型须不断吸收新能力而不遗忘旧能力。

现有方法的两难困境：
1. **共享参数策略**：所有任务共用同一参数集 → 异构任务竞争参数 → **灾难性遗忘**
2. **专用模块策略**：每个新任务分配独立模块 → 不存在干扰 → **参数爆炸**（任务流越长，参数越多）

> X is insufficient: Shared parameters cause forgetting; dedicated modules per task cause parameter explosion. Neither scales over long task streams.

### 设计 (Design — CRAM)
CRAM（Centroid Routing Adaptive MoE）结合两种机制：

1. **Centroid Routing（质心路由）**：
   - 将每个任务的样本特征聚类，取质心（centroid）作为任务代表
   - 路由时，将新样本分配给最近质心对应的专家模块
   - 任务之间自然隔离，避免干扰

2. **Adaptive Rank MoE（自适应秩 MoE）**：
   - 评估当前专家能力与新任务需求之间的**能力差距（capability gap）**
   - 根据差距大小动态分配**参数秩（rank）**：差距大 → 分配高秩 LoRA；差距小 → 低秩
   - 避免对简单任务过度分配参数，整体参数效率高于固定秩方案

---

## 故事弧 / Story Arc

> *"Multimodal models need to continuously learn new tasks without forgetting old ones. Shared parameters cause forgetting; dedicated modules inflate model size over long task streams. CRAM resolves this by routing tasks to task-specific experts (preventing interference) and allocating expert capacity adaptively based on the capability gap (preventing parameter waste)."*

---

## 创新点 / Innovation

1. **质心路由**：用聚类质心而非可学习路由门控路由，计算更稳定，且无需额外监督标签
2. **自适应秩分配**：能力差距驱动的秩分配是 MCIT 领域的新思路，与固定秩 MoE 相比参数更节省
3. **无灾难性遗忘 + 参数可控**：CRAM 首次在 MLLM 领域同时实现这两点

与相关工作比较：
| 方法 | 干扰防止 | 参数效率 | 自适应分配 |
|------|----------|----------|----------|
| Shared LoRA | ❌ | ✅ | ❌ |
| Task-specific LoRA | ✅ | ❌ | ❌ |
| MoCE / BranchLoRA | 部分 | 部分 | ❌ |
| **CRAM** | ✅ | ✅ | ✅ |

---

## 关键指标 / Key Metrics

| 比较对象 | CRAM 结果 |
|----------|-----------|
| Shared parameters | Better on all MCIT tasks, no forgetting |
| Dedicated modules per task | Comparable performance, significantly fewer parameters |
| Fixed-rank MoE | Same performance, lower parameter count via adaptive rank |

---

## 打分明细 / Scoring Breakdown

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 质心路由+自适应秩是MCIT新组合，设计优雅 |
| Experimental SOTA Delta | 10 | 15 | 在MCIT上超过共享参数和专用模块方案；指标数值有限 |
| Experimental Quality/Ablations | 10 | 15 | 多任务实验；消融信息有限 |
| Efficiency | 9 | 10 | 自适应秩显著节省参数；部署成本低 |
| Generalization | 4 | 5 | 跨多个多模态指令任务 |
| Domain Relevance | 10 | 25 | 持续学习新电商品类/审核规则场景，但距直接应用还有gap |
| **Total** | **65** | **100** | |

---

## 与电商/内容治理的关联

- **电商 MLLM 的持续迭代**：商品品类、审核规则（节假日特价活动规则、直播新规等）频繁更新，MLLM 需要持续学习新能力而不遗忘旧有知识 — CRAM 提供了一种可行方案
- **跨品类商品理解**：服装 → 电子 → 美妆等新品类上线时，CRAM 可按需分配专家容量
- **多模态内容审核规则迭代**：新型违规内容出现时（如新形式的劣质商品图），通过 CRAM 快速扩展审核能力而不影响既有模块
