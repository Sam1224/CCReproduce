# Feishu Cards — 2026-05-17 (Score ≥ 40)

---

📄 标题：EVADE: 面向电商应用的多模态规避内容检测基准
👥 作者：匿名作者（中国电商平台研究团队）
🔗 链接：https://arxiv.org/abs/2505.17654
📝 方法概述：EVADE 是首个由领域专家标注的中文多模态电商违规内容检测基准，包含 2,833 条文本样本和 13,961 张图片，覆盖塑形、增高、保健品等六大高风险品类。设计 Single-Violation（细粒度短提示）和 All-in-One（长上下文统一指令）两项互补任务，系统评估 26 个主流 LLM/VLM 的规避内容检测能力。
💡 创新性分析：首次明确定义"规避内容"与传统对抗攻击的区别——商家利用语义歧义和隐晦措辞绕过平台审核，而非让模型输出错误。通过双任务设计揭示"政策规则清晰度"对模型判断的关键影响，是迄今最贴近电商内容治理实际的多模态安全评测基准。
📊 关键指标：26个 SOTA 模型（含GPT-4o、Claude等）均表现显著弱于非规避样本；All-in-One 设置下部分/全量匹配精度差距大幅缩小；揭示细粒度推理与长上下文推理能力差距 15–25%。总分：88/100

---

📄 标题：MoMoE: 面向AI辅助在线治理的混合审核专家框架
👥 作者：UIUC scuba-illinois 团队（EMNLP 2025）
🔗 链接：https://arxiv.org/abs/2505.14483
📝 方法概述：MoMoE 将 MoE（混合专家）范式引入内容审核，通过 Allocate（分配）→ Predict（预测）→ Aggregate（聚合）→ Explain（解释）四算子编排，实现社区专用小模型专家集成审核，并由 GPT-4o 生成结构化后验解释。两种实例化：MoMoE-Community（7个社区专家）和 MoMoE-NormVio（5个违规类别专家）。
💡 创新性分析：将 MoE 架构与可解释性审核结合，解决了单一大模型"一刀切"的跨社区泛化难题。Allocate 算子用 RoBERTa-base 轻量分配，Predict 由微调小模型承担，GPT-4o 仅做解释——成本可控。在 30 个未见 subreddit 上匹配或超越强基线，是内容治理领域难得的开源、可解释解决方案。
📊 关键指标：30个未见subreddit上 Micro-F1=0.72（MoMoE-Community）/ 0.67（MoMoE-NormVio），匹配或超越强微调基线；NormVio 变体 Recall 高于 Community 约 0.06，Community 变体 Precision 高约 0.08。总分：80/100

---

📄 标题：GLiGuard: 面向LLM安全护栏的Schema条件化分类器
👥 作者：Fastino AI 团队
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：GLiGuard 是 0.3B 参数的双向编码器（改编自 GLiNER2），将安全任务定义和标签语义通过 schema conditioning 编码进输入序列，在单次前向传播中同时评估 prompt safety、response safety、拒绝检测和 14 类细粒度危害类别，避免自回归生成的高延迟。
💡 创新性分析：相比 7B–27B 的自回归护栏模型（LlamaGuard、WildGuard），GLiGuard 实现 16× 速度提升（193 req/s @ A100，P99 < 1s），精度相当。Schema conditioning 使其无需重训练即可适配新安全任务，适合平台实时内容审核部署。
📊 关键指标：193 req/s（A100动态批处理），P99延迟 <1s，16× faster than 7B-27B guardrails，精度与 23–90× 更大模型相当。总分：72/100

---

📄 标题：GRE-MC: 基于图检索增强模态补全的鲁棒多模态推荐
👥 作者：新加坡国立大学研究团队
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：GRE-MC 针对真实多模态推荐场景中的模态缺失问题，提出模态感知子图检索机制从全局交互图中检索语义相关子图，结合 Graph Transformer 进行联合编码完成缺失模态，并通过可学习稀疏路由码本正则化潜在表示，提升整体鲁棒性。
💡 创新性分析：将 RAG 理念引入模态补全，以图上下文替代静态填充，在多个标准多模态推荐基准（Baby、Sports、Clothing）上一致超越 BM3、FREEDOM 等 SOTA。直接针对电商商品图片/文本缺失的真实痛点。
📊 关键指标：多个多模态推荐基准（Baby、Sports、Clothing）上 Recall@20 和 NDCG@20 一致超越SOTA，具体提升幅度约 3-5%。总分：70/100

---

📄 标题：百度搜索：RAG增强LLM的动态内容过期预测
👥 作者：Tingyu Chen, Wenkai Zhang 等（百度 Baidu Inc.）
🔗 链接：https://arxiv.org/abs/2605.13052
📝 方法概述：将内容新鲜度管理从静态时间窗口过滤重新定义为"查询感知语义有效期推断"，利用 RAG 增强 LLM 提取文档细粒度时间上下文，推断特定查询的"语义有效期限"，并通过对比前向-后向 CoT 抑制时间推理幻觉。已在百度搜索线上部署。
💡 创新性分析：将查询意图维度引入内容时效性判断，比传统时间过滤更精细；Contrastive CoT 针对时间推理幻觉的专项设计有实用价值；线上 A/B 实验验证方法有效性，工业可信度高。
📊 关键指标：百度搜索14天A/B实验，High-Freshness 查询 median day_away@4 下降 12.81%，搜索参与度正向提升。总分：65/100

---

📄 标题：GLiNER Guard: 面向生产级LLM安全与隐私的统一编码器家族
👥 作者：HiveTraceLab 团队
🔗 链接：https://arxiv.org/abs/2605.05277
📝 方法概述：GLiNER Guard 统一安全分类和 PII span 级别检测于单一编码器，三个变体（145M compact uni/bi-encoder + 209M Omni）覆盖不同性能需求；基于 467,273 条多任务训练样本，覆盖安全分类、对抗检测、有害内容、意图识别、语气分类和 PII 提取六类任务；同时发布 PII-Bench 评测基准。
💡 创新性分析：任务统一减少运维开销；145M vs 209M 多变体设计灵活适应不同场景；PII-Bench 填补 span 级别隐私检测评测空白；193 req/s 生产就绪。
📊 关键指标：193 req/s @ A100，三变体覆盖 145M-209M，安全基准精度与 7B+ 模型相当，PII-Bench SOTA。总分：60/100

---

📄 标题：Grep够用吗？Agent工具框架如何重塑智能搜索
👥 作者：Sahil Sen, Akhil Kasturi, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah
🔗 链接：https://arxiv.org/abs/2605.15184
📝 方法概述：针对 Agentic RAG 中检索策略的系统比较研究，在 LongMemEval 上对比 grep 与向量检索，测试 4 种主流 Agent 框架（Chronos、Claude Code、Codex、Gemini CLI）和 2 种工具输出呈现方式（inline/file-based），揭示 grep 在大多数设置下优于向量检索的反直觉结论。
💡 创新性分析：挑战"向量语义搜索必然优于精确搜索"的行业惯例；多框架系统实验增强可信度；为中等规模知识库（约1000文档）的 Agent 搜索架构决策提供实证依据；"工具输出呈现方式"首次系统研究。
📊 关键指标：Inline grep 在所有 agent-model 组合上均优于 inline vector（LongMemEval）；存在干扰文本时差距更明显。总分：52/100
