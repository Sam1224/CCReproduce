# PHF: Beyond Isolated Behaviors — Hierarchical User Modeling for LLM Personalization

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **arXiv** | https://arxiv.org/abs/2606.02300 |
| **提交日期** | 2026-06-02（公告窗口 2026-06-03~04 GMT+8） |
| **作者** | Liang Wang, Xinyi Mou, Xiaoyou Liu, Tiannan Wang, Yuqing Wang, Zhongyu Wei |
| **机构** | 未公开披露（搜索 snippet 推断） |
| **领域标签** | `User Modeling` `LLM Personalization` `Hierarchical` `Sociology-Inspired` `Recommendation` |
| **桶位** | WEAK |

---

## 方法概述

现有 LLM 个性化方法以"扁平行为（flat behavioral paradigm）"聚合用户行为序列，缺乏对行为如何组织成深层结构的显式建模，导致**行为孤立（isolated behaviors）**问题：单条行为被作为独立信号处理，无法反映用户的稳定偏好或群体共性。

PHF（Practice-Habitus-Field）借鉴社会学家布尔迪厄的实践理论，将 LLM 个性化重构为三层次层级框架：

1. **Practice（实践）**：用户的单次行为，如点击、购买、评论——对应个体行为序列。
2. **Habitus（惯习）**：用户行为的时间积累，形成稳定的个人倾向——对应用户长期偏好画像。
3. **Field（场域）**：跨相似用户的共享规律——对应群体级别的用户聚类与共性偏好。

三层次相互对应于三个计算组件：行为序列编码（Practice）、时序积累偏好建模（Habitus）、跨用户相似度聚合（Field），最终综合三层信号指导 LLM 的个性化输出。

---

## 故事弧线 / Story Arc

**LLM 个性化只聚合行为序列，忽视行为的层次组织结构** → PHF 从社会学的"实践-惯习-场域"理论出发，把用户建模分解为行为→稳定倾向→群体规律三层次 → 三层次联合指导 LLM 个性化，超越扁平化 behavior aggregation。

---

## 创新性分析

- **社会学理论迁移**：把布尔迪厄的实践论引入 LLM 个性化是新颖的跨学科视角，框架设计有充分的理论基础。
- **三层次分解**：Practice/Habitus/Field 与 RecSys 中的 session/user/community 分层思想对应，但通过 LLM 框架统一建模，适配 open-domain 个性化。
- **Field（跨用户共性）**：引入群体级别的相似用户信息，类似 collaborative filtering 的思想，在纯 LLM 框架中较为少见。
- **局限**：三层次的计算实现可能引入较大开销；"场域"的聚类边界需要预定义。

---

## 关键指标

| 数据集 | 任务 | PHF vs. Baseline |
|--------|------|-----------------|
| 个性化基准（多任务） | 文本个性化 | 三层次 ablation 各有贡献 |

- 具体指标未从搜索 snippet 获取，待原论文。

---

## 评分 (满分 100)

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 20 / 30 | 社会学理论迁移新颖；计算实现是已有技术组合 |
| 实验指标 | 9 / 15 | 未获具体数字 |
| 实验质量 | 10 / 15 | 三层次 ablation；多任务覆盖 |
| 效率 | 6 / 10 | 三层次计算有额外开销 |
| 泛化性 | 4 / 5 | 框架通用于 open-domain 个性化 |
| 相关性 | 13 / 25 | 达人粉丝用户建模、推荐个性化间接相关 |
| **Total** | **62** |
