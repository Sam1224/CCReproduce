# 2026-06-07 Paper 巡检记录

本日巡检的结构化结果已固化在 `papers.json`，用于后续自动化汇总与 Web 展示。

## 本日精选（含评分）

- **82/100** — OpAI-Bench (arXiv:2606.06481)：渐进式人→AI 编辑轨迹 + 多粒度溯源，用于评测 AI 文本检测器在混合作者区间的非单调失效模式。
- **80/100** — AdaPlanBench (arXiv:2606.05622)：双约束（世界+用户）渐进披露的交互式规划基准，评测 LLM agent 的自适应重规划与约束归纳能力。
- **79/100** — OneReason (arXiv:2606.06260)：生成式推荐推理范式（感知→认知 + 三层 CoT + RL 配方），报告线上显著收益。
- **79/100** — camroll-agent / Camera Roll VQA (arXiv:2606.05275)：个人视觉记忆长时序检索 + agent，展示高召回与极低 token 成本。
- **72/100** — Code2LoRA (arXiv:2606.06492)：超网络生成 repo 级 LoRA，包含演化版（diff→GRU→adapter）以对抗代码漂移。
- **61/100** — Stance Simulation Audit (arXiv:2606.06443)：反事实上下文修订审计框架，用于评估 LLM 立场模拟对上下文的敏感性。

## 评分机制（百分制）

总分 = 方法创新性(30) + 实验指标(15) + 实验质量(15) + 方法效率(10) + 方法泛化性(5) + 论文相关性(25)。

## 代码复现说明（>=80）

- OpAI-Bench：作者已提供完整代码仓库（GitHub: VILA-Lab/OpAI-Bench），无需重复实现。
- AdaPlanBench：作者已提供完整代码与数据集入口（GitHub: JiayuJeff/AdaPlanBench），无需重复实现。

> 注：若后续发现代码仓库为空/缺少核心实现，将在本目录下新增对应的 toy 复现工程。