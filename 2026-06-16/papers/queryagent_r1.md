# QueryAgent-R1: E-Commerce Query Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation |
| **作者** | Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng |
| **机构** | Alibaba International Digital Commercial Group |
| **链接** | https://arxiv.org/abs/2606.05671 |
| **arXiv ID** | 2606.05671 |
| **提交日期** | June 4, 2026 |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**中文：**  
来自阿里巴巴国际商业的QueryAgent-R1提出了一个记忆增强的智能体框架，用于解决电商搜索中查询推荐（Query Recommendation）与商品转化率脱节的问题。传统方法仅优化查询相关性（CTR），却忽视下游商品与用户偏好的匹配度（CVR）。QueryAgent-R1通过**链式检索优化（Chain-of-Retrieval Optimization）**，让智能体在生成推荐查询后立即执行真实库存检索，并根据检索结果验证和优化查询。结合智能体强化学习过程中的一致性奖励（Consistency Reward），同时优化查询相关性和下游商品转化。

**English:**  
Alibaba International proposes QueryAgent-R1, a memory-augmented RL agent for e-commerce query recommendation that bridges the generation-retrieval gap. Existing methods optimize query-level CTR while ignoring downstream product CVR. QueryAgent-R1 uses chain-of-retrieval: the agent generates query candidates, executes real inventory retrieval, validates/refines queries based on retrieved products, and receives a consistency reward that jointly optimizes CTR and CVR via RLVR.

---

## 故事弧线 / Story Arc

**现有方法不足 →** 查询推荐只优化点击率（CTR），生成的查询在电商搜索结果页上匹配的商品与用户真实意图不符，导致转化率（CVR）低。  
**本文设计 →** 智能体在生成查询后即刻用真实库存执行检索验证，通过一致性奖励联合优化CTR+CVR，实现端到端对齐。

---

## 创新性 / Innovation

1. **链式检索优化**：将真实库存检索融入查询生成循环，使生成过程具备"接地"能力。
2. **记忆增强智能体**：维护历史查询-检索-转化记忆，避免重复生成低效查询。
3. **一致性奖励**：通过RL一致性奖励信号同步优化查询相关性和商品转化，是业内首次端到端对齐。
4. **RLVR框架**：将可验证奖励（RLVR）引入电商查询推荐，强化可靠的生成决策。

**与前工作的差异：** 传统查询推荐基于纯文本相关性；本文首次将真实检索结果作为奖励信号融入生成过程。

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | 结果 | 基线 |
|-------------|------|------|------|
| 阿里巴巴国际电商平台 | CTR (Click-Through Rate) | 提升 | 传统查询推荐 |
| 阿里巴巴国际电商平台 | CVR (Conversion Rate) | 提升 | 传统查询推荐 |
| 在线A/B测试 | 端到端商业指标 | 显著正向 | 现有方法 |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 链式检索+RLVR一致性奖励，电商场景新颖 |
| 实验SOTA增量 (SOTA Delta) | 11 | 15 | 在线实验CTR+CVR双提升 |
| 实验质量/消融 (Exp Quality) | 11 | 15 | 真实生产实验 |
| 效率 (Efficiency) | 7 | 10 | RLVR训练开销较大 |
| 泛化性 (Generalization) | 3 | 5 | 主要针对电商搜索场景 |
| 领域相关性 (Domain Relevance) | 23 | 25 | 直接针对电商搜索/推荐核心问题 |
| **Total** | **77** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.05671
