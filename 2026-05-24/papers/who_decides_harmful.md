# Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **Title** | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework |
| **Authors** | Ewelina Gajewska et al. |
| **Affiliation** | (Not recovered from search; European academic/industry lab inferred) |
| **arXiv** | [2605.01416](https://arxiv.org/abs/2605.01416) |
| **Submitted** | 2 May 2026 |
| **Domain** | Content Governance · Multi-Agent LLM · Personalized Moderation |
| **Bucket** | STRONG |
| **Code** | No official code |

---

## 分数 / Score

| 维度 | 分数 | 满分 |
|---|---|---|
| Innovation | 20 | 30 |
| Experimental SOTA delta | 11 | 15 |
| Experimental quality / ablations | 9 | 15 |
| Efficiency | 5 | 10 |
| Generalization | 3 | 5 |
| Domain relevance (ecom + governance) | 20 | 25 |
| **Total** | **68** | **100** |

**Justification**: The idea of personalizing content moderation to user sensitivity profiles using multi-agent LLM orchestration is genuinely novel and practically important. The 32% accuracy improvement over one-size-fits-all moderation is significant. Domain relevance is high: influencer content governance increasingly needs to account for different user demographics and sensitivities. Multi-agent overhead is a practical limitation. Experimental depth could be stronger.

---

## 方法概述 / Method Summary

当前内容审核系统采用"一刀切"的政策，无法适应不同用户群体对敏感内容的差异化感知（例如：同样的暴力描述，成年用户和未成年用户的可接受阈值完全不同；文化背景不同的用户对冒犯性内容的判断也差异巨大）。这种刚性政策导致过度屏蔽（误杀合规内容）和审核不足（放行敏感内容）并存。

**多智能体个性化内容审核框架**：

1. **专家智能体层 (Expert Agents)**  
   - 多个领域专家 LLM，各自专注于特定风险类型（仇恨言论、成人内容、虚假信息、暴力等）。  
   - 每个专家基于明确的政策定义对内容给出独立评估。

2. **管理智能体 (Manager Agent)**  
   - 协调内容分析流程：路由内容到相关专家、综合各专家评估、解决冲突判断。  
   - 动态选择激活哪些专家（降低不必要的计算开销）。

3. **幽灵档案智能体 (Ghost Profile Agent)**  
   - 模拟目标用户视角：基于用户人口统计学、历史敏感性记录、地域文化背景构建"虚拟用户档案"。  
   - 将用户视角注入审核决策，实现个性化判断。

**Story Arc**: 一刀切内容审核导致误伤和漏审 → 通过专家-管理-幽灵档案三层多智能体框架，将用户个人敏感性融入审核政策，实现准确率提升 32% 的个性化内容治理。

---

## 创新性分析 / Innovation Analysis

**与 prior work 的对比**:
- 传统规则引擎：静态规则，无法适应用户差异。
- 单一 LLM 审核（如 OpenAI Moderation API）：统一判断，缺乏个性化。
- **本文创新**：  
  (a) **Ghost Profile Agent** 是核心创新：通过"模拟用户视角"而非依赖用户历史数据来实现个性化，保护隐私；  
  (b) 专家-管理-档案三层分离使系统模块化，可独立更新各层；  
  (c) 在内容审核领域将 RLHF 的"个人偏好对齐"思路迁移到实际审核决策中。

**局限**: 多智能体调用成本高，实时审核场景需要优化；Ghost Profile 构建准确性难以验证。

---

## 关键指标 / Key Metrics

| 实验设置 | 指标 | 本框架 | 基线 | 提升 |
|---|---|---|---|---|
| 个性化内容审核 | Accuracy | 显著提升 | 统一政策基线 | **+32%** |
| 用户敏感性对齐 | Precision/Recall | 改善 | 静态规则 | 方向性提升 |

---

## 与电商内容生态的关联

- **达人内容分级审核**: 不同年龄段用户使用电商平台时，对同一达人营销内容（如医美、减肥产品）的敏感性不同。个性化审核可以在平台侧实现差异化内容分级。
- **多国合规**: 全球电商平台面临不同国家/地区的内容法规差异，Ghost Profile Agent 的文化背景建模直接适用此场景。
- **用户投诉降低**: 准确率提升 32% 意味着更少的误屏蔽（商家内容被错误下架）和更少的漏审（违规内容未被发现）。

---

## 论文链接

- arXiv: https://arxiv.org/abs/2605.01416
