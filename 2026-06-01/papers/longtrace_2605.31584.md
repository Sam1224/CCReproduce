# LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards |
| **Authors** | Nianyi Lin, Jiajie Zhang, Lei Hou, Juanzi Li |
| **Affiliation** | Tsinghua University (THU-KEG group) |
| **Venue** | arXiv preprint (HuggingFace Daily Papers 2026-06-01) |
| **arXiv** | https://arxiv.org/abs/2605.31584 |
| **GitHub** | https://github.com/THU-KEG/LongTraceRL |
| **Submitted** | 2026-05-29 |

---

## 方法概述 / Method Summary

LLM 在长上下文推理中常陷入"迷失在中间"困境——面对大量干扰文档时，无法定位并整合关键证据。RLVR（可验证奖励强化学习）已展现潜力，但现有方法两大不足：(1) 干扰文档区分度低（confusability 不足）；(2) 只有结果级别的稀疏奖励，无法监督中间推理步骤。

**LongTraceRL** 的两项核心贡献：

1. **分层干扰文档构建（Tiered Distractor Construction）**：利用搜索 Agent 轨迹构建两类干扰文档——Agent 读过但未引用的文档（高混淆度）、Agent 仅检索未打开的文档（低混淆度）。通过知识图谱随机游走生成多跳问题，使干扰文档更贴近真实检索场景。
2. **Rubric 奖励（细粒度过程奖励）**：以推理链上每个中间实体（gold entity）为过程监督信号，仅对最终答案正确的响应应用 positive-only 策略，避免错误推理链上的噪声反向传播。

**故事弧线：** 长上下文 RLVR 因干扰文档区分度不足、奖励信号稀疏而难以训练 → LongTraceRL 以搜索 Agent 轨迹构建高质量分层干扰文档，以实体级 rubric 奖励提供密集过程监督，在 4B–30B 模型和 5 个长上下文基准上一致优于强基线。

---

## 关键指标 / Key Metrics

| Dataset | Models | Result |
|---------|--------|--------|
| 5 long-context benchmarks | 4B, 7B, 30B LLMs | Consistently outperforms strong baselines |

---

## 评分 / Score

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | 分层干扰文档 + rubric 实体级过程奖励，在 RLVR 长上下文方向有明确贡献 |
| Experimental SOTA Delta | 12 | 15 | 5 基准一致超越，3 种规模模型验证 |
| Experimental Quality / Ablations | 12 | 15 | 严谨的消融，Tsinghua KEG 组高质量工作 |
| Efficiency | 7 | 10 | 标准 RL 训练，干扰文档构建开销合理 |
| Generalization | 4 | 5 | 多模型规模和多基准 |
| Domain Relevance (ecom + governance) | 10 | 25 | 与 RAG / 搜索 Agent 相关，可迁移到电商多跳问答，但非直接场景 |
| **Total** | **67** | **100** | |
