# Unintended Consequences of Recommender System Interventions: Evidence from a Field Experiment

## 基本信息 / Basic Info

| Field | Details |
|-------|---------|
| **Title** | Unintended Consequences of Recommender System Interventions: Evidence from a Field Experiment |
| **Authors** | Shilei Luo, Song Yao, Dennis J. Zhang |
| **Affiliation** | Olin Business School, Washington University in St. Louis |
| **ArXiv** | [2606.08265](https://arxiv.org/abs/2606.08265) |
| **Submitted** | June 6, 2026 |
| **Domain Tags** | `platform-governance` `recommendation-systems` `field-experiment` `content-distribution` |
| **Total** | **71 / 100** |

---

## 故事弧线 / Story Arc

**问题：** 平台内容干预（如"睡眠提醒"等用户行为引导活动）通常被视为静态"nudge"来评估，忽视了推荐系统会从干预引发的用户行为中自适应学习，导致干预效果与预期相反。

**解决方案：** 通过大规模短视频平台现场实验，揭示"强制探索机制"（Forced-Exploration Mechanism）：干预暴露了对被推广内容的高潜在需求，触发推荐算法更新，反而强化了原本试图缓解的参与循环。

**Story arc:** Static intervention assumptions are insufficient → Field experiment reveals algorithm feedback loops amplify opposite effects.

---

## 方法概述 / Method

研究设计了一个大规模现场实验，在某短视频平台的"睡眠提醒"活动中进行：
- 实验规模：大规模用户随机分配（控制组 vs 干预组）
- 干预：夜间使用时段向用户推送睡眠提醒，旨在减少深夜使用
- 观测：干预前后及干预后数周的用户平台使用行为

**关键发现：**
- 干预组深夜参与度增加 **14.75%**（而非减少）
- 整体平台使用量增加 **2.18%**
- 效果在实验结束后数周持续存在

**机制解释（强制探索机制）：**
1. 干预强制平台向用户推送了原本算法不会推荐的内容类型（深夜内容）
2. 这揭示了用户对该类内容的高潜在需求
3. 推荐算法从用户正向反馈中学习，更新了推荐策略
4. 算法更新强化了深夜参与循环，与干预目标相反

---

## 创新性分析 / Innovation

**核心洞察：** 平台治理中的内容干预不是静态 nudge，而是给推荐算法提供了探索新策略的机会，可能产生持久的系统性改变。

**对平台治理的意义：**
- 达人/内容治理政策需要考虑算法反馈循环
- 单次干预的效果评估框架需要包含算法自适应维度
- 违规内容整治可能引发算法重新分配推荐流量

---

## 关键指标 / Key Metrics

| Setting | Metric | Result |
|---------|--------|--------|
| Sleep reminder campaign | Late-night engagement | +14.75% (opposite effect) |
| Sleep reminder campaign | Overall platform usage | +2.18% |
| Post-experiment persistence | Effect duration | Weeks after intervention ended |

---

## 评分明细 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | Forced-exploration mechanism insight is novel and important for platform governance |
| Experimental SOTA delta | 10 | 15 | 14.75% engagement increase is striking and statistically significant |
| Experimental quality / ablations | 14 | 15 | Large-scale field experiment with rigorous causal identification |
| Efficiency | 3 | 10 | Not an efficiency paper |
| Generalization | 4 | 5 | Generalizable insight to all algorithmic recommendation platforms |
| Domain relevance | 18 | 25 | Platform governance, recommendation algorithm policy — relevant but not direct ecom product |
| **Total** | **71** | **100** | |

---

## 中文摘要

本文通过短视频平台大规模现场实验研究推荐系统干预的意外后果。一项旨在减少深夜使用的"睡眠提醒"活动，却使深夜参与度增加了 14.75%，整体平台使用量增加了 2.18%，效果在活动结束后数周仍持续存在。研究提出"强制探索机制"解释这一反常现象：干预强迫推荐算法探索新内容策略，暴露了用户对深夜内容的高潜在需求，导致算法更新后强化了本应被缓解的参与循环。对于电商内容生态的启示：内容治理政策（如达人内容整治）不仅是静态 nudge，更是给推荐算法提供探索机会，可能引发算法重新分配推荐流量，产生持久影响。
