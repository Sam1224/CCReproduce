# Feishu Cards — 2026-06-01

---

📄 标题：Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching
👥 作者：Wei Chee Yew, Hailun Xu, Sanjay Saha, Xiaotian Fan, Hiok Hian Ong et al.
🔗 链接：https://arxiv.org/abs/2512.03553
📝 方法概述：提出 HybridMod 混合直播内容审核框架，双管道并行——有监督分类管道（处理已知违规）+ MLLM 驱动相似性匹配管道（处理新兴违规），MLLM 以知识蒸馏方式赋能两路轻量推理，满足实时部署需求。
💡 创新性分析：MLLM 同时充当教师（蒸馏给分类器）和特征提取器（相似性管道），明确分离已知/未知违规两类问题；与单一端到端模型相比，更易迭代维护；首篇在生产 A/B 测试中验证的直播多模态审核框架。
📊 关键指标：生产环境 @ 80% Precision：分类管道 Recall 67%，相似性管道 Recall 76%；大规模 A/B 测试不良直播观看量下降 6–8%。（KDD 2026 接收）

---

📄 标题：Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment
👥 作者：Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He
🔗 链接：https://arxiv.org/abs/2601.16027
📝 方法概述：提出 CS-VAR（跨会话证据感知检索增强检测器）。训练阶段：LLM 对当前会话 + RAG 检索到的历史会话证据进行联合推理，识别跨场次重复恶意模式，生成结构化风险判断；知识蒸馏将 LLM 推理能力迁移至轻量领域模型。推理阶段：小模型独立完成实时风险评分，无需调用 LLM。
💡 创新性分析：首次将 RAG 范式引入直播风险检测，显式建模跨会话复发模式；"训练重推理轻"的蒸馏设计满足工业实时需求；输出可定位风险信号支持人工审核。
📊 关键指标：大规模工业数据集线下 SOTA + 线上正向验证（2025 年电商平台 5 万余主播被处罚场景背景）。

---

📄 标题：Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation (TGQ-Former)
👥 作者：Yufei Guo, Jing Ma, Tianlu Zhang (清华大学); Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong (京东); Jungong Han (清华大学)
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：提出 TGQ-Former，以商品结构化元数据（标题、品类、属性）为语义锚点引导 Q-Former 视觉 token 提取。Hybrid-Query Connector 分为元数据锚定查询（过滤促销贴片和背景噪声）和探索性查询（捕获补充视觉线索）两路，Reliability-Aware Dual-Gated Vector Modulation 动态融合两路特征，应对噪声图像输入。
💡 创新性分析：首次针对电商噪声商品图像场景设计文本引导的双路 Q-Former 连接器；可靠性门控自适应融合相比固定权重更鲁棒；针对 I2I 检索任务（而非生成）优化，直接对齐工业场景。
📊 关键指标：大规模电商真实数据（全量检索）H@100 平均提升 6.04%，一致优于强连接器基线和端到端 MLLM。

---

📄 标题：Adapting Vision-Language Models for E-commerce Understanding at Scale
👥 作者：Matteo Nulli, Vladimir Orshulevich, Tala Bazazo, Christian Herold, Michael Kozielski et al.
🔗 链接：https://arxiv.org/abs/2602.11733
📝 方法概述：大规模实验研究，系统探讨如何将通用 VLM 适配到电商场景（属性中心推理、多图聚合、卖家噪声内容鲁棒性），同时保留通用多模态能力，避免灾难性遗忘。
💡 创新性分析：首个系统研究电商 VLM 适配策略的实证工作，覆盖多主流架构；提供可复用的适配配方，工程实践价值高。
📊 关键指标：多下游电商任务显著提升（具体数值见论文），通用基准上保留原有性能。

---

📄 标题：Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion (GRE-MC)
👥 作者：Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He（新加坡国立大学）
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：提出 GRE-MC，以模态感知子图检索从全图中获取语义相关上下文，再通过联合编码图变换器（全局注意力）完成缺失模态重建，可学习稀疏路由码本正则化潜在 embedding。
💡 创新性分析：突破现有模态补全方法局限于局部邻居的瓶颈，子图检索显著扩大语义感受野；图变换器联合编码避免两阶段误差累积。
📊 关键指标：多模态推荐基准上持续超越 SOTA（具体数值见论文）。

---

📄 标题：LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards
👥 作者：Nianyi Lin, Jiajie Zhang, Lei Hou, Juanzi Li（清华大学 THU-KEG）
🔗 链接：https://arxiv.org/abs/2605.31584
📝 方法概述：提出 LongTraceRL，利用搜索 Agent 轨迹构建分层干扰文档（高/低混淆度），并以推理链中间实体为 rubric 细粒度过程奖励，通过 RLVR 训练 LLM 长上下文推理能力。
💡 创新性分析：分层干扰文档提高 RLVR 训练挑战性；实体级过程奖励填补稀疏奖励的训练空洞；开源实现。
📊 关键指标：5 个长上下文基准上 4B/7B/30B 模型一致超越强基线。

---

📄 标题：COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation
👥 作者：Tianyi Zhou, Dongrui Liu, Leitao Yuan, Jing Shao, Xia Hu
🔗 链接：https://arxiv.org/abs/2605.31264
📝 方法概述：从专家操作轨迹（traces）自动蒸馏出可检查、可纠错的结构化 AI 技能，注入通用 LLM Agent，使其表现出特定专家的知识和行为模式。
💡 创新性分析：端到端 trace-to-skill 蒸馏框架，首次实现专家知识的可移植、可审计包装；可拓展到电商专家标注知识的 Agent 化。
📊 关键指标：详见论文；框架已验证于多种专家知识蒸馏场景。
