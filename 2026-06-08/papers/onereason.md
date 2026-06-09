# OneReason Technical Report

## 基本信息 / Basic Info

| Field | Detail |
|-------|--------|
| **Title** | OneReason Technical Report |
| **Authors** | OneRec Team (83 contributors; lead: Kuaishou AI team) |
| **Affiliation** | Kuaishou Technology |
| **arXiv ID** | [2606.06260](https://arxiv.org/abs/2606.06260) |
| **Submitted** | ~5 June 2026 (indexed in 8 June 2026 listing) |
| **Bucket** | STRONG |
| **Related Work** | OneRec (base), OneRec-Think (2510.11639), OpenOneRec (2512.24762) |

---

## 方法概述 / Method Summary

**问题背景 / Problem**: 生成式推荐模型（如 OneRec 家族，已在快手短视频/直播/广告/电商全场景落地）能从规模扩展中受益，但其**推理能力（Reasoning）难以激活**——因为 item token 序列无法构成有意义的 Chain-of-Thought（CoT），不像自然语言那样可以进行逐步推理。

**Story Arc**: *"生成式推荐只能靠规模，无法像 LLM 一样 'think before answer' → 探索在 item-space 中注入推理的方法（OneRec-Think）→ 发现 thinking mode 对比 non-thinking mode 无显著优势 → 本报告分析原因并提出新研究方向"*

**方法 / Method**:

OneReason 是对 OneRec 家族推理能力系统性探索的技术报告：

1. **OneRec Baseline**:
   - 端到端生成式推荐：输入用户行为序列 → 直接输出推荐 item token
   - 已大规模部署：短视频、直播、广告、电商

2. **OneRec-Think（已有工作）**:
   - **Itemic Alignment**: item token 与语义文本锚定，建立 item→语义映射
   - **Reasoning Scaffolding**: 在 item token 序列间插入自然语言推理步骤
   - **Recommendation-Specific Reward**: 定制 RL 奖励函数

3. **关键发现（OneReason 报告）**:
   - Thinking mode ≈ Non-thinking mode（无显著提升）
   - Item token 稀疏语义导致 CoT 质量不佳
   - 但 OneRec-Think 在部分场景仍有 0.159% App Stay Time 提升（快手线上）

4. **新研究方向（报告提出）**:
   - 混合 item-text 推理链
   - 用户兴趣语义化建模
   - 多 Agent 推荐推理

**实验指标（OneRec-Think 线上）**:
```
快手线上 A/B 实验：
  App Stay Time:  +0.159%
  Watch Time:     +0.169%
  Video View:     +0.150%
  Follows:        +0.431%
  Forwards:       +0.758%
```

---

## 故事弧 / Story Arc

- **Insufficient**: LLM 领域"think before answer"范式大幅提升推理质量，但生成式推荐的 item-token 空间无法构造有效 CoT
- **We Design**: OneRec-Think（Itemic Alignment + Reasoning Scaffolding + 定制 Reward）尝试激活推理能力
- **Finding**: Thinking mode 并未明显优于 Non-thinking mode，首次公开这一负向结论
- **Path Forward**: OneReason 报告揭示根本原因并提出混合推理路线图

---

## 创新分析 / Innovation

| 维度 | 分析 |
|------|------|
| **负向结论的价值** | 首个系统报告"reasoning 在生成式推荐中不 work"的工业级实验，对社区有重要指导意义 |
| **规模经济** | OneRec 家族已覆盖短视频/直播/广告/电商全场景，复现条件丰富 |
| **Itemic Alignment** | item token 语义化对齐是推荐系统与 LLM 融合的关键桥梁 |
| **开源生态** | OpenOneRec 已开源（GitHub: Kuaishou-OneRec/OpenOneRec），便于学术跟进 |

---

## 关键指标 / Key Metrics

| 场景 | Metric | OneRec-Think | Baseline (OneRec) |
|------|--------|--------------|-------------------|
| Kuaishou 快手线上 | App Stay Time | +0.159% | — |
| Kuaishou 快手线上 | Watch Time | +0.169% | — |
| Kuaishou 快手线上 | Follows | +0.431% | — |
| Amazon Beauty (offline) | Recall@5 | 0.0563 | 0.0460 (base) |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 20 | 负向发现价值高；但核心 OneRec-Think 工作此前已有预告 |
| SOTA Delta | 15 | 9 | 线上指标正向但幅度小；核心发现是 reasoning 无显著优势 |
| Experimental Quality | 15 | 10 | 快手工业级实验，多指标追踪，但 ablation 细节分散在多篇 |
| Efficiency | 10 | 7 | 讨论了推理链的计算开销问题 |
| Generalization | 5 | 5 | 短视频/直播/广告/电商全覆盖 |
| Domain Relevance | 25 | 22 | 短视频/直播电商推荐场景，高度相关 |
| **Total** | **100** | **73** | 重要工业报告，尤其对生成式推荐研究方向有参考价值 |
