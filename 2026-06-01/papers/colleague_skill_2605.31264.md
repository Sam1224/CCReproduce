# COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation |
| **Authors** | Tianyi Zhou, Dongrui Liu, Leitao Yuan, Jing Shao, Xia Hu |
| **Affiliation** | (arXiv: cs.AI + cs.CL + cs.LG) |
| **Venue** | arXiv preprint (HuggingFace Daily Papers 2026-06-01) |
| **arXiv** | https://arxiv.org/abs/2605.31264 |
| **Submitted** | 2026-05 |

---

## 方法概述 / Method Summary

LLM Agent 需要携带特定专家的知识、判断风格和交互习惯，以构建"以人为本"的 Agent。但可行动知识通常嵌入在异构的操作轨迹（traces）中，而非结构化指令，导致 Agent 难以直接学习。现有记忆和 Persona 系统只能捕捉片段证据；技能框架提供可移植的包装格式，但缺乏端到端的自动化蒸馏流程。

**COLLEAGUE.SKILL** 提出：将专家操作轨迹（traces）自动转化为可检查、可纠错、可 Agent 使用的"技能（skill）"表示，具体流程为：
1. **Trace 解析**：解析专家操作记录，识别关键决策点和推理模式。
2. **技能蒸馏（Skill Distillation）**：LLM 从 traces 中提炼出结构化技能定义，包括触发条件、执行步骤、预期结果。
3. **可检查性设计**：技能以可读、可验证的格式存储，支持人工审核和纠错。
4. **技能注入 Agent**：蒸馏得到的技能可注入到通用 LLM Agent 中，使其表现出专家行为模式。

**故事弧线：** 专家知识散落在不可结构化的轨迹数据中，无法直接赋能 LLM Agent → COLLEAGUE.SKILL 端到端自动蒸馏专家轨迹为可移植技能，为以人为本的 Agent 构建提供了新框架。

---

## 评分 / Score

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | trace-to-skill 端到端蒸馏框架新颖，可检查性设计有价值 |
| Experimental SOTA Delta | 9 | 15 | 摘要中具体数字未完全披露 |
| Experimental Quality / Ablations | 9 | 15 | 有限细节 |
| Efficiency | 7 | 10 | 蒸馏框架本身较轻量 |
| Generalization | 3 | 5 | 通用场景声明，但实验细节待考察 |
| Domain Relevance (ecom + governance) | 12 | 25 | 与数据标注、知识蒸馏相关，可迁移到电商专家知识提取，但非直接场景 |
| **Total** | **60** | **100** | |
