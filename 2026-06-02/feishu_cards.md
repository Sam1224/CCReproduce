# Feishu Cards — 2026-06-02

> 所有评分 ≥ 40 的论文，每篇生成一个飞书卡片。共 5 篇。

---

📄 标题：PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning
👥 作者：Yusong Zhao, Yuejin Xie, Youliang Yuan, Junjie Hu, Jitian Guo, Yujiu Yang, Pinjia He（中国香港中文大学深圳、清华大学）
🔗 链接：https://arxiv.org/abs/2606.02443
📝 方法概述：PaSBench-Video 构建了包含 740 段流式视频（481 风险 + 259 无风险）的基准，覆盖驾驶、医疗、日常生活、工业生产四大场景。核心创新在于"主动预警"评测范式：在危险发生前的窗口期内，模型必须在仅能观察历史帧的条件下，发出时间校准且内容准确的告警。评测指标逐级递进：检测命中率 → 时间校准命中率 → 内容正确性。
💡 创新性分析：全球首个面向流式视频主动安全告警的基准，填补了现有评测体系中"时间精度"和"误报率"两大空白。与 StreamingBench 等相比，PaSBench-Video 要求模型在因果约束下进行时序推理而非事后分析，更贴近真实内容治理部署场景（如直播违规预警）。
📊 关键指标：13 个 MLLM 在最严格指标上均未超过 20%；Recall 与误报率 Pearson 相关系数达 0.64，揭示当前模型依赖表面场景线索而非真正理解风险。评分：73/100（STRONG 领域相关）

---

📄 标题：ExpWeaver: LLM Agents Learn from Experience via Latent RAG
👥 作者：Tao Feng, Tianyang Luo, Jingjun Xu, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You
🔗 链接：https://arxiv.org/abs/2606.01041
📝 方法概述：ExpWeaver 将 LLM Agent 的经验存储与检索从显式文本空间移至模型自身的隐状态（latent space）。每个解码步骤均执行潜在空间向量检索，通过跨注意力+门控残差将历史经验融入当前生成过程，整条 pipeline 用强化学习端到端优化。无需独立 RAG 模块，显著减少 token 开销。
💡 创新性分析：全球首次在 LLM 内部 hidden state 层面实现逐解码步的经验检索与融合；与 ExpeL、Self-RAG 等文本 RAG 方案相比，架构更紧凑、检索与生成联合优化、无额外参数开销。13 类多样任务（含推荐、推理、编程）上 12/13 达 SOTA，平均领先基线 >6.8%。
📊 关键指标：13 多样任务基准（QA/推理/编程/推荐等），SOTA on 12/13 tasks，平均 >6.8% 超越基线。评分：72/100（WEAK 可迁移）

---

📄 标题：RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift
👥 作者：Kilho Son, Paul Hsu, Cha Zhang, Dinei Florencio（微软）
🔗 链接：https://arxiv.org/abs/2606.01697
📝 方法概述：RCEM 通过知识蒸馏将 LLM 的查询改写能力迁移至 embedding 模型：训练时以改写查询嵌入为锚点对齐对话查询嵌入，推理时无需额外 LLM 改写。训练无需对话-文档相关性标注，只需（对话, 改写）对即可。在 QReCC、TopiOCQA、TREC CAsT 三个基准上均超越强基线，分布偏移场景下 Recall@10 最高提升 20%。
💡 创新性分析：将 LLM 改写能力蒸馏至嵌入层而非轻量 LM 是新颖角度；免去推理时改写调用，显著降低延迟；对分布偏移的针对性设计在实际电商搜索中（节日/日常场景切换、PC/移动端差异）有重要价值。
📊 关键指标：QReCC 分布偏移测试 Recall@10 +20%；TopiOCQA、TREC CAsT 一致性提升。评分：66/100（WEAK 可迁移）

---

📄 标题：CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning
👥 作者：Jun-Tao Tang, Zhen-Hao Xie, Yu-Cheng Shi, Da-Wei Zhou
🔗 链接：https://arxiv.org/abs/2606.02502
📝 方法概述：CRAM 解决多模态持续指令微调（MCIT）中的两难困境——共享参数导致灾难性遗忘，专用模块导致参数爆炸。核心设计：(1) 质心路由（Centroid Routing）将任务样本聚类并分配给最近质心专家，自然隔离任务间干扰；(2) 自适应秩 MoE（Adaptive Rank MoE）根据现有专家能力与新任务需求的差距动态分配 LoRA 秩，避免参数浪费。
💡 创新性分析：质心路由比可学习门控更稳定、不需要任务标签；自适应秩分配是 MCIT 领域的新方向；两者结合首次在 MLLM 连续任务流上同时实现"无遗忘+参数可控"。适用于电商 MLLM 随新品类/新审核规则持续迭代的场景。
📊 关键指标：在多个 MCIT 任务上优于共享参数基线（无遗忘）；参数量低于专用模块基线（效率）；自适应秩低于固定秩 MoE。评分：65/100（WEAK 可迁移）

---

📄 标题：Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events
👥 作者：Xiaolin Liu, Yilun Zhu, Xiangyu Zhao, Xuehui Wang, Yan Li, Xin Li, Haoyu Cao, Xing Sun, Shaofeng Zhang, Xu Yang, Zhihang Zhong, Xue Yang
🔗 链接：https://arxiv.org/abs/2606.02522
📝 方法概述：Moment-Video 是专为诊断视频 MLLM"瞬时视觉保真度"设计的基准，包含 1000 条 QA 对（7 场景/25 子类别），聚焦仅持续几帧的关键事件。4 类任务：时间发生判断、时间计数、动作描述、时间推理。每题必须依赖局部短暂证据作答，不能靠全局场景或语言先验蒙混过关。
💡 创新性分析：首个专注采样敏感"瞬时事件"的视频 MLLM 基准；揭示稀疏采样、视觉 token 压缩等常用技术对瞬时证据的破坏；33 个模型（含 Seed-2.0-Pro/GPT-5.4/Gemini）全面评估揭示显著性能鸿沟（最优仅 39.6%）。对短视频内容审核（关键违规动作往往只有几帧）有直接启示。
📊 关键指标：Seed-2.0-Pro 39.6%（最优）；GPT-5.4 26.9%；大多数开源模型 <25%；计数/时间推理任务最难。评分：64/100（STRONG 领域相关）
