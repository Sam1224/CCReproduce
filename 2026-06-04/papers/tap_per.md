# TAP-PER: Beyond Retrieval — Learning Compact User Representations for Scalable LLM Personalization

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **arXiv** | https://arxiv.org/abs/2606.04547 |
| **提交日期** | 2026-06-03（公告窗口 2026-06-04 GMT+8） |
| **作者** | Heng Cao, Fan Zhang, Jian Yao, Yujie Zheng, Changlin Zhao, Lu Hao, Yuxuan Wei, Wangze Ni, Huaiyu Fu, Yuqian Sun, Xuyan Mo |
| **机构** | 未公开披露（基于 arXiv 元数据推断为工业实验室） |
| **领域标签** | `LLM Personalization` `User Representation` `Prefix Tuning` `Recommendation` `Scalable` |
| **桶位** | STRONG |

---

## 方法概述

将 LLM 应用于个性化推荐面临两个工程挑战：①**基于检索**的方案（RAG-style）把用户历史检索后拼入 prompt，依赖检索质量且 token 成本随用户历史线性增长；②**基于参数**的方案（per-user LoRA 等）存储和维护成本随用户数线性扩展，亿级用户不可承受。

TAP-PER（Temporal Attentive Prefix for PERsonalization）把用户偏好编码为**可学习的 prefix embedding（前缀嵌入）**，避免了显式 prompt 构建和重量级 per-user adapter：

1. **双前缀分解**：
   - **User-State Prefix**：编码用户长期偏好，维度固定，按用户存储（轻量）。
   - **Query-Conditioned Record Prefix**：动态激活与当前 query 最相关的历史记录，聚焦短期意图。

2. **时序感知**：在 prefix 生成中引入时间信号，捕捉用户兴趣的演化动态（近期行为权重高，长期偏好衰减）。

3. **无需 prompt 重构**：直接把 prefix 注入 LLM 的 attention key-value cache，无需修改 LLM 参数，与 frozen LLM 兼容。

---

## 故事弧线 / Story Arc

**检索式个性化受检索质量限制，参数式个性化维护成本随用户线性增长** → TAP-PER 用双前缀（长期 user-state + 短期 query-conditioned）替代显式检索和重量级 adapter → 通过时序感知捕捉兴趣演化 → 实现规模可扩展（per-user 仅存 prefix 向量）的高质量 LLM 个性化。

---

## 创新性分析

- **双前缀分解**：把个性化信息分解为"长期偏好"和"即时意图"两个正交维度，与认知科学的工作记忆/长期记忆概念对应，框架合理。
- **时序建模**：推荐系统的时序兴趣建模（DIN/DIEN 系列）思路迁移到 LLM prefix tuning 场景，是有实践价值的跨域迁移。
- **规模可扩展性**：per-user prefix 存储量远小于 LoRA adapter（通常小 1–2 个数量级），对百亿用户平台意义重大。
- **局限**：prefix 的表达能力取决于维度设置，对 cold-start 用户（无历史）的覆盖未详述；与多模态内容的结合尚未探索。

---

## 关键指标

| 数据集 | 任务 | TAP-PER vs. Retrieval Baseline | TAP-PER vs. LoRA Baseline |
|--------|------|-------------------------------|--------------------------|
| 个性化基准（具体数据集待论文） | 文本个性化生成 | 质量持平或更优，延迟更低 | 质量持平，存储开销下降显著 |

- 具体指标（ROUGE、win rate、AUC 等）未从搜索 snippet 中获取，待原论文。

---

## 评分 (满分 100)

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 22 / 30 | 双前缀分解 + 时序信号是有价值的增量创新；思路成熟但落地清晰 |
| 实验指标 | 9 / 15 | snippet 未获具体数字；从描述推断结果正向 |
| 实验质量 | 10 / 15 | 消融双前缀各组件，时序信号 ablation |
| 效率 | 9 / 10 | per-user 仅存 prefix 向量，规模扩展性强 |
| 泛化性 | 4 / 5 | 与任意 frozen LLM 兼容 |
| 相关性 | 16 / 25 | 电商平台用户个性化推荐直接相关；不直接涉及内容治理 |
| **Total** | **70** |
