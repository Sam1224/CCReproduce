# Feishu Cards — 2026-06-11 Daily Inspection
# Papers with score ≥ 40

---

📄 标题：UNIVID: Unified Vision-Language Model for Video Moderation（统一视觉语言模型视频内容审核）
👥 作者：Kejuan Yang, Yizhuo Zhang et al.（ByteDance，字节跳动）
🔗 链接：https://arxiv.org/abs/2606.05748
📝 方法概述：提出以"策略感知字幕"（Policy-Aware Caption）为核心的统一 VLM 视频审核流水线，三阶段级联：（A）风险过滤器融合字幕和视觉特征；（B）审核执行器包含 UNIVID-Lite（轻量决策）和 UNIVID-RAG（历史违规召回）；（C）趋势治理模块通过缓存嵌入和趋势头微调自适应检测新兴风险。
💡 创新性分析：核心创新在于将可解释字幕作为中间表示，使审核决策可被人工验证，并支持多任务复用；不同于传统黑盒分类器，统一表示大幅降低系统维护成本；UNIVID-RAG 借助历史违规事件提升召回。
📊 关键指标：生产部署于 ByteDance 平台；字幕策略对齐带来可解释性和多任务复用；趋势头微调支持快速适应新兴违规内容，无需全量重训（分数：84/100）

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation（链式检索强化学习电商查询推荐）
👥 作者：Alibaba International Digital Commerce Group（阿里巴巴国际数字商业集团）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：提出记忆增强的 Agentic 框架，将商品库检索结果嵌入 RL 训练循环中：Agent 生成查询后即调用真实检索系统验证和精化，设计一致性奖励同时优化查询相关性和商品转化对齐，并用兴趣图谱高效管理用户长期记忆。
💡 创新性分析：首次将商品库实时检索结果作为 RL 奖励信号的一部分，解决现有方法高 CTR 低 CVR 的核心矛盾；GRPO 端到端训练避免两阶段优化的不一致；兴趣图谱抽象解决长序列用户历史的计算瓶颈。
📊 关键指标：工业数据集 CTR+CVR 联合提升；部署于日活数千万的大型电商平台；公开数据集 SOTA（分数：82/100）

---

📄 标题：Unintended Consequences of Recommender System Interventions（推荐系统干预的意外后果：来自现场实验的证据）
👥 作者：Shilei Luo, Song Yao, Dennis J. Zhang（华盛顿大学奥林商学院）
🔗 链接：https://arxiv.org/abs/2606.08265
📝 方法概述：通过某短视频平台"睡眠提醒"活动的大规模现场实验，研究平台内容干预对推荐算法的反馈效应，揭示"强制探索机制"：干预暴露了用户对被推广内容的高潜在需求，触发算法更新并反向强化目标行为。
💡 创新性分析：首次用因果推断方法量化内容干预→算法反馈→用户行为的因果链；强制探索机制解释了为什么善意干预可能产生相反效果；对平台达人/内容治理政策的制定具有重要指导意义。
📊 关键指标：深夜参与度+14.75%（干预目标相反）；整体平台使用量+2.18%；效果持续数周（分数：71/100）

---

📄 标题：EVID-Bench: When Seeing Is Not Believing — A Benchmark for Search-Grounded Video Misinformation Detection（基于搜索的视频虚假信息检测基准）
👥 作者：多机构合作团队
🔗 链接：https://arxiv.org/abs/2606.04098
📝 方法概述：提出 EVID-Bench，包含 222 个视频、9种操纵类型（AI生成、单源剪辑、多源拼接），所有样本经人工验证确认前沿 MLLM 仅凭视觉检查无法检测；要求系统通过网络搜索相关视频并跨视频比对来识别虚假信息。
💡 创新性分析：提出搜索增强的跨视频比对检测范式，填补了现有基准仅支持单视频检测的空白；揭示了当前最先进模型的根本局限性；对平台内容安全团队的检测能力建设有重要指引。
📊 关键指标：222 个验证样本；9种操纵类型；前沿 MLLM 单视频检测精度≈随机（分数：70/100）

---

📄 标题：Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance（基础模型先验赋能的主动学习：类别不均衡下的高效学习）
👥 作者：Jiancheng Zhang, Meiqing Li, Qi Zhang, Yinglun Zhu（UC Riverside; CMU; WPI）
🔗 链接：https://arxiv.org/abs/2606.07630
📝 方法概述：利用 CLIP 等基础模型的零样本能力提供类别先验，与任务小模型协同做不均衡感知的主动学习决策，同时内嵌噪声标签过滤机制，解决数据标注中噪声和类别不均衡的联合挑战。
💡 创新性分析：首个系统研究噪声+不均衡联合问题的主动学习框架；基础模型先验无需额外标注即可提供可靠类别分布估计；对电商商品属性标注、内容审核标注等大规模标注场景直接适用。
📊 关键指标：图像和文本双域大幅降低标注成本；少数类精度显著提升（分数：64/100）

---

📄 标题：Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge（缓解多模态 LLM 评判模型中的感知判断偏差）
👥 作者：Seojeong Park et al.（KAIST）
🔗 链接：https://arxiv.org/abs/2606.02578
📝 方法概述：发现 MLLM-as-Judge 在视觉与文本线索冲突时存在感知判断偏差，通过最小化视觉修改构建反事实判断数据集，结合 GRPO 奖励和批次排序训练修正偏差（ICML 2026）。
💡 创新性分析：系统性发现和量化 MLLM 评判的感知偏差；反事实数据集构建方法提供可验证的监督信号；对自动化内容质量评估和标注质控系统有直接改进价值。
📊 关键指标：ICML 2026；感知正确性 SOTA；判断一致性显著提升（分数：60/100）

---

📄 标题：Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models（可恢复视觉 Token 路由）
👥 作者：Cheng-Yu Yang, Shao-Yuan Lo, Yu-Lun Liu
🔗 链接：https://arxiv.org/abs/2606.12412
📝 方法概述：现有 VLM Token 剪枝方法永久移除 Token 导致信息不可恢复。提出可恢复路由机制：Token 被路由到辅助路径而非丢弃，在需要时可恢复，实现效率与信息保留的平衡（2026-06-11 提交）。
💡 创新性分析：将"移除"改为"路由"的视角转换在 VLM 效率领域具有启发性；为电商内容理解 VLM 的大规模部署提供更安全的效率方案。
📊 关键指标：VQA 基准优于 Token 剪枝；推理 FLOPS 与剪枝方法相当；代码开源（分数：59/100）

---

📄 标题：Context-Driven Incremental Compression for Multi-Turn Dialogue Generation（多轮对话上下文增量压缩）
👥 作者：Yeongseo Jung et al.（HKUST, NVIDIA 等）
🔗 链接：https://arxiv.org/abs/2606.12411
📝 方法概述：C-DIC 将对话分解为语义线程，在紧凑记忆中维护可修订的逐线程压缩状态，通过 retrieve→revise→write-back 轻量级循环实现跨轮信息共享和过期记忆修正（ICML 2026，2026-06-11 提交）。
💡 创新性分析：线程级增量压缩比整体对话摘要更精细；可修订机制解决了跨轮记忆失效问题；对电商客服 Agent 等长对话场景有实用价值。
📊 关键指标：ICML 2026；记忆效率显著降低；优于截断和摘要方法（分数：58/100）

---

📄 标题：InSemRAG: Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking（意图感知检索与语义保留分块的高效 RAG）
👥 作者：多机构合作
🔗 链接：https://arxiv.org/abs/2606.01240
📝 方法概述：InSemRAG 框架通过意图感知检索器（分解为意图子查询）和语义保留 chunking（基于语义边界切分）解决传统 RAG 意图无关检索和信息碎片化问题。
💡 创新性分析：两个改进的组合解决 RAG 基本问题；语义边界 chunking 对知识密集型电商 FAQ 系统有实用价值；适用于平台政策知识库检索。
📊 关键指标：RAG 基准优于朴素 RAG；语义完整性提升（分数：56/100）
