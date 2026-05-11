# EKTM: Effective Knowledge Transfer for Multi-Task Recommendation Models

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | Effective Knowledge Transfer for Multi-Task Recommendation Models |
| **arXiv ID** | [2605.05730](https://arxiv.org/abs/2605.05730) |
| **提交日期** | 2026-05-07 |
| **作者** | Guohao Cai, Jun Yuan, Zhenhua Dong |
| **机构** | (商业平台，工业级推荐系统团队) |
| **领域桶** | STRONG |
| **综合评分** | **82 / 100** |

---

## 方法概述 (Chinese)

电商平台的转化率（CVR）预测模型因用户转化行为稀疏而面临数据不足问题。现有多任务学习方法往往存在负迁移（negative transfer）现象——来自不相关任务的知识反而损害目标任务性能。

EKTM（Effective Knowledge Transfer for Multi-Task Recommendation Models）提出了一种跨任务知识有效迁移框架：
1. **Router 模块**：整合并在各任务间分发共享知识，动态路由相关信息；
2. **Transmitter 模块**：每个 CVR 任务独立配备，将 Router 输出转化为适合当前任务的知识表示；
3. **Enhanced 模块**：确保迁移的知识在增强目标任务的同时，不损害原任务学习效果。

该方法在多个基准数据集上超越现有 SOTA，并在商业平台完成在线 A/B 测试，两个主流量场景均已全量部署。

## Method Overview (English)

CVR ranking models suffer from sparse conversion data. Existing multi-task learning methods risk negative transfer. EKTM introduces a Router module that integrates and distributes cross-task knowledge, a per-task Transmitter module that adapts router knowledge to each CVR task, and an Enhanced module ensuring transferred knowledge truly benefits the target task. Evaluated on benchmark datasets and validated via online A/B test on a commercial platform with full deployment.

---

## Story Arc

**传统多任务推荐学习在任务差异大时容易发生负迁移 → EKTM 设计 Router-Transmitter-Enhanced 三模块架构，动态路由并精准迁移跨任务知识，同时保护原任务学习不受干扰。**

> Prior MTL methods treat all tasks as equally transferable, causing negative transfer when task distributions diverge. EKTM routes knowledge through a learnable router, transforms it via per-task transmitters, and guards target task integrity via an enhancement module.

---

## 创新性分析

1. **任务自适应知识路由**：Router 不是简单参数共享，而是动态感知任务相关性后路由；
2. **Transmitter 的任务特化**：每个 CVR 任务独立的 Transmitter 解决了多任务干扰问题；
3. **Enhanced 模块的保护机制**：防止迁移知识"稀释"原始任务信号，是精细工程设计的体现；
4. **工业级验证**：A/B 测试和全量上线是学术界罕见的强验证，具备极高可信度。

**与先前工作的差异**：相比 MMOE、PLE 等经典多任务架构，EKTM 专注于 CVR 稀疏性下的跨任务知识转化路径，而非仅共享底层表示。

**新颖性可信度**：高。方法设计合理，工业部署结果是最强验证。

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | EKTM | 最佳基线 |
|---|---|---|---|
| 商业平台在线A/B测试 | eCPM提升 | **+3.93%** | — |
| 商业平台 | 部署范围 | 两个主流量场景全量 | — |
| 多基准数据集 | AUC / GAUC | SOTA | 次优方法 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 20 | Router-Transmitter-Enhanced 架构设计扎实，非革命性但切实解决负迁移 |
| 实验SOTA增益 (SOTA delta) | 15 | 13 | 工业A/B测试 +3.93% eCPM，多基准SOTA |
| 实验质量与消融 (Quality) | 15 | 13 | 工业A/B测试 + 消融实验，验证充分 |
| 效率 (Efficiency) | 10 | 7 | 增加模块但工业可接受 |
| 泛化性 (Generalization) | 5 | 4 | 两个主流量场景验证 |
| 领域相关性 (Domain) | 25 | 25 | 核心电商推荐，CVR建模，已部署 |
| **总分** | **100** | **82** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.05730
- HTML版: https://arxiv.org/html/2605.05730
- 代码复现: `../code/EKTM/`
