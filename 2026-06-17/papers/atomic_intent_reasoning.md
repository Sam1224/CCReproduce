# Atomic Intent Reasoning (AIR): Bringing LLM Semantics to Industrial Cross-Domain Recommendations

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations |
| **Authors** | Zhuohang Jiang, Yuxin Chen, Shijie Wang, Haohao Qu, Jindong Zhou, Wenqi Fan, Qing Li, Dongxu Liang, Jun Wang |
| **Affiliations** | The Hong Kong Polytechnic University; Kuaishou Technology |
| **arXiv** | [2606.10357](https://arxiv.org/abs/2606.10357) |
| **Submitted** | ~2026-06-10 |
| **Domain Tags** | e-commerce, cross-domain recommendation, LLM, knowledge distillation, industrial |
| **Code** | `code/AIR/` |

---

## 方法概述 / Method Summary

现有工业级跨域推荐系统难以同时满足 LLM 语义理解与在线低延迟需求——直接在线调用 LLM 成本过高、延迟无法接受。AIR（Atomic Intent Reasoning）将 LLM 推理迁移至离线阶段，通过"原子意图"（Atomic Intent）将用户行为分解为细粒度、可组合的语义意图单元；在线阶段通过高效检索与组合这些意图向量，动态构建用户跨域意图表示，无需在线 LLM 调用，实现约 400× 推理加速。最终集成到快手电商（Kuaishou E-commerce）大规模 Multi-Task Learning 排序模型中，并通过线上 A/B 实验验证。

**Story arc**: 跨域推荐（内容消费域→电商转化域）的语义鸿沟使得现有协同过滤方法缺乏语义迁移能力，直接在线部署 LLM 又有延迟瓶颈 → 设计离线 LLM 推理 + 原子意图检索组合机制，将 LLM 知识注入在线排序系统。

**Key components**:
1. **Offline Atomic Intent Mining**: 利用 LLM 对用户历史行为（内容观看、搜索）进行离线推理，提取原子意图向量并存入索引库
2. **Online Intent Composition**: 在线阶段仅做轻量检索（ANN）+ 线性组合，构建用户当前意图表示
3. **Cross-domain Transfer**: 将内容侧意图向量映射至电商排序特征，注入 MTL 排序模型
4. **Semantic Consistency Preservation**: 通过对比学习确保离线原子意图与在线组合意图的语义一致性

---

## 创新性分析 / Innovation Analysis

**vs. prior work**:
- 相比直接 LLM 在线服务（延迟高、成本不可控），AIR 通过离线预计算实现 400× 加速，是迄今工业部署中最具扩展性的 LLM 推荐方案之一
- 相比传统跨域推荐（数据迁移、领域适配），AIR 从语义层面而非行为层面桥接内容域与电商域，泛化能力更强
- "原子意图"的粒度设计允许任意组合，避免了语义空间爆炸问题

**Novelty assessment**: 创新具体可行，离线 LLM 推理 + 在线原子组合的范式在工业界有较强的参考价值；A/B 实验结果具有说服力。

---

## 关键指标 / Key Metrics

| Dataset/System | Metric | AIR | Baseline |
|---------------|--------|-----|----------|
| Kuaishou E-commerce (online A/B) | GMV | **+3.446%** | — |
| Kuaishou E-commerce (online A/B) | Multiple core metrics | significant improvement | — |
| Inference | Acceleration vs. online LLM | **~400×** | 1× |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 26 | 30 | 离线 LLM 推理 + 原子意图组合，工业部署创新 |
| Experimental SOTA delta | 14 | 15 | 线上 A/B 实验 +3.446% GMV，高可信 |
| Experimental quality / ablations | 14 | 15 | 大规模 A/B 实验，多核心指标验证 |
| Efficiency | 10 | 10 | 400× 加速，工业可部署 |
| Generalization | 4 | 5 | 跨域推荐泛化，但仅在 Kuaishou 验证 |
| Domain relevance | 23 | 25 | 快手电商、达人内容到电商转化，直接相关 |
| **Total** | **91** | **100** | 工业 A/B 实验+400× 加速，顶尖电商推荐论文 |
