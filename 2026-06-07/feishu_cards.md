# Feishu Cards — 2026-06-07 Daily Paper Inspection

Papers with score ≥ 40 (sorted by score descending)

---

📄 标题：QueryAgent-R1: 基于检索接地的电商查询推荐 Agent RL 框架
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang et al.（阿里巴巴国际数字商业集团）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：提出 QueryAgent-R1，一种"记忆增强 + 检索接地"的 agentic RL 框架用于电商查询推荐。核心创新是 Chain-of-Retrieval 循环：agent 生成 query → 在真实商品库执行检索 → 基于检索结果反思并精炼 query，形成端到端的"查询-商品"对齐优化。训练中引入 Consistency Reward 同时优化 CTR 与 CVR 代理信号，打破两者对立；记忆模块（Interest Graph）压缩用户长期交互历史为高效画像。
💡 创新性分析：首次将真实商品库检索结果纳入 query 推荐的 RL 奖励信号，彻底解决了"高 CTR 低 CVR"的优化目标错位问题。与纯语义 query 生成方法相比，QueryAgent-R1 的奖励信号与实际业务指标（转化率）高度对齐，工程落地价值极高。
📊 关键指标：在线 A/B 测试（阿里巴巴国际电商平台）| Query CTR +2.9% | Guided CVR +3.1% | 离线多基准均优于强基线

---

📄 标题：OpAI-Bench：多粒度 AIGC 文本检测基准（渐进式人→AI 编辑轨迹）
👥 作者：Sondos Mahmoud Bsharat, Jiacheng Liu, Zhiqiang Shen et al.（阿联酋 MBZUAI + UCL）
🔗 链接：https://arxiv.org/abs/2606.06481
📝 方法概述：提出 OpAI-Bench，将"修订轨迹"而非独立文档对作为 benchmark 基本单元。从人类原稿出发，按 5 类 AI 编辑操作（润色/改写/风格重写/压缩/扩写）和固定覆盖比例生成 v0–v8 累积修订版本，同时保留文档/句子/Token/Span 四粒度溯源标注。在 4 个领域评测 17 类 AI 文本检测器（含 LLM-as-detector），揭示"非单调可检测性"现象：中等 AI 覆盖+压缩操作往往比全 AI 生成更难检测。
💡 创新性分析：核心创新是把"修订轨迹"作为 benchmark 一等公民，使 AI 覆盖率、操作类型、修订历史三个变量可控分析。"非单调可检测性"是对现有检测器能力边界的重要实证揭示，对平台 AIGC 治理策略设计有直接指导意义（如：不能只盯"全 AI 生成"文本，混合写作中的压缩段落更危险）。
📊 关键指标：检测器数量：文档级×8 + 句子级×7 + 细粒度×2 | reports 上 Fast-DetectGPT F1-AI: v1=61.3 → v8=29.1（性能随修订反而下降）| essays 上 Gemini-Flash: 峰值 v3=71.5，但 v1 仅 9.5

---

📄 标题：AdaPlanBench：双约束渐进披露下的 LLM Agent 自适应规划基准
👥 作者：Jiayu Liu, Cheng Qian, Heng Ji et al.（UIUC）
🔗 链接：https://arxiv.org/abs/2606.05622
📝 方法概述：提出 AdaPlanBench，307 个家庭任务的多轮交互规划基准，同时包含世界约束（资源/环境）和用户约束（偏好/禁忌）。约束不在开始时全部给出，而是在 agent 提出违规计划后才揭示，迫使 agent 在多轮中累积约束记忆并反复修订。使用可扩展的双约束构建流水线为每个任务生成约束画像，在 10 个主流 LLM 上评测。
💡 创新性分析：将"约束归纳"和"约束跟踪"从规划能力的隐含前提提升为可量化的评测维度。违约驱动的渐进揭示机制比"一次性给定所有约束"更贴近真实 agent 部署场景（如：平台合规 agent 在执行过程中逐步遇到政策约束）。
📊 关键指标：307 个 household 任务 | 10 个主流 LLM 评测 | 最佳模型准确率仅 67.75% | 约束累积越多性能越差 | 用户约束比世界约束更难

---

📄 标题：OneReason：激活生成式推荐中的推理能力（快手×字节达到线上显著收益）
👥 作者：OneRec Team (83+ authors)（快手 Kuaishou Inc. + 字节跳动 ByteDance）
🔗 链接：https://arxiv.org/abs/2606.06260
📝 方法概述：Technical Report 揭示"生成式推荐中 thinking-mode 为何失败"并给出解决方案。核心问题：item token 缺乏语义感知，无法构造有意义的 CoT。解决路径：(1) 感知增强预训练（item token 与自然语言描述对齐）；(2) 三层认知增强 CoT（用户意图→候选评估→推荐决策）；(3) Specialize-then-Unify RL 训练；(4) Fast-Slow Thinking 在线架构（慢路处理高价值请求，27.2% retrieval share）。
💡 创新性分析：首次明确诊断"推荐 item token 语义空洞→thinking-mode 无效"这一根本原因，并提供系统性解决方案。Fast-Slow 双路在线架构直接响应工业部署中的效果-延迟 Pareto 前沿，是可落地的架构范式。
📊 关键指标：线上 A/B（快手多业务）| Conv +12.643% | Click +1.709% | CVR +1.865% | CTR +0.519% | 低活用户 Revenue +13.323%

---

📄 标题：camroll-agent：个人相册 VQA 的 AI Agent（层级记忆 + 极低 token 成本）
👥 作者：Thao Nguyen, Krishna Kumar Singh, Donghyun Kim, Yong Jae Lee, Yuheng Li（U. Wisconsin-Madison + Korea Univ. + Adobe Research）
🔗 链接：https://arxiv.org/abs/2606.05275
📝 方法概述：研究个人相册（camera roll）上的 VQA 问题：assistant 需在跨年、数万张个人照片的长时序视觉流中检索证据回答问题。构建 camroll 数据集（50 用户、31,476 张图、2,500 QA 对），提出 camroll-agent：层级记忆+最小工具集的对话式 agent，迭代式精确检索将 token 成本从 750K 压缩至约 3.2K（234×）。
💡 创新性分析：将"个人视觉记忆"作为独立研究对象，层级记忆（事件级摘要+图片级 metadata）设计可迁移至电商场景（达人历史内容检索、用户视觉购买历史理解）。token 效率量化（234× 压缩）为大规模多模态 agent 部署提供工程参考。
📊 关键指标：MC Recall=88.5 / Acc=70.5 | Free-form Judge=4.11/5 | Token ~3,200 vs. baseline ~750,000 | 234× token 压缩

---

📄 标题：Code2LoRA：超网络生成仓库级 LoRA（含演化版对抗代码漂移）
👥 作者：Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie（University of Waterloo）
🔗 链接：https://arxiv.org/abs/2606.06492
📝 方法概述：用超网络将整个代码仓库编码为固定向量，一次性生成仓库专属 LoRA 适配器，实现 0 推理 token 额外开销的仓库知识注入。演化版（Code2LoRA-Evo）用 GRU 累积 commit diff 的隐状态，增量更新 adapter 以应对持续演化的代码库。同时构建 RepoPeftBench（604 个 Python 仓库）。
💡 创新性分析：将超网络 LoRA 从短文本/单文档条件扩展到全仓库条件，并提出演化感知的增量更新机制，解决仓库知识随 commit 漂移的问题。技术适用性高，可作为治理 LLM 工程化工作流的基础设施。
📊 关键指标：静态轨道: Cross-repo EM=63.8%, In-repo EM=66.2% | 演化轨道: Cross-repo EM=60.3%, vs 共享 LoRA +5.2pp

---

📄 标题：'Your AI Text is not Mine'：基于现实假设的 AIGC 检测重定义与评测（AITDNA Benchmark）
👥 作者：Nils Dycke, Marina Sakharova, Nico Daheim, Iryna Gurevych（TU Darmstadt）
🔗 链接：https://arxiv.org/abs/2606.04906
📝 方法概述：系统定义多种 AIGC 文本概念（全生成/AI 辅助写作/AI 改写等），收集真实人机共同创作文本，标注完整 genesis 信息（edit history + AI 交互历史），构建 AITDNA Benchmark。揭示现有检测器在"现实混合创作"场景下的普遍失效。
💡 创新性分析：定义先行——厘清"什么是 AIGC"的共识性框架，避免 benchmark 和检测器建立在隐含且不一致假设上。与 OpAI-Bench 互补：OpAI-Bench 关注受控渐进修订，AITDNA 关注真实多样创作模式。两者结合为平台内容治理提供更完整的 AIGC 检测评测基础。
📊 关键指标：1,821 段真实 Reddit 对话 | 完整 genesis 标注 | 原始立场模拟 Acc=77.64, Macro-F1=78.10 | 现有检测器在真实混合创作场景下性能显著下降

---

📄 标题：Revising Context, Shifting Simulated Stance：反事实上下文修订审计 LLM 立场模拟
👥 作者：Xinnong Zhang, Wanting Shan, Hanjia Lyu, Zhongyu Wei, Jiebo Luo（复旦大学 + 罗切斯特大学）
🔗 链接：https://arxiv.org/abs/2606.06443
📝 方法概述：提出"反事实上下文修订"审计框架：在原对话上下文中模拟用户立场，通过受控文本或多模态（meme）修订改变上下文后重新模拟，测量立场转移率与方向性偏移，审计 LLM 立场模拟对上下文变化的敏感性。数据：1,821 段 Reddit 对话（涉及 DeepSeek/Claude/Llama）。
💡 创新性分析：为"LLM 作为社会意见模拟器"提供系统性审计工具，引入 meme 等多模态上下文扩展。对电商平台的舆情分析、达人观点识别和内容偏见审计具有方法参考价值。
📊 关键指标：1,821 段 Reddit 对话 | 基础模拟: Acc=77.64, Macro-F1=78.10 | 上下文修订显著触发立场转移
