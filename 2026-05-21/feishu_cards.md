# 飞书卡片 — 2026-05-21 日报 (分数 ≥ 40 的论文)

---

📄 标题：TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation
👥 作者：(工业电商研究团队，arXiv 2605.17366，2026-05-17)
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：提出 TGQ-Former，将商品结构化元数据（类目/品牌/属性）作为语义引导，通过混合查询 Q-Former 将视觉 token 提取分为"元数据锚定流"和"自由探索流"两路，并引入可靠性感知双门控向量调制模块（RDGVM），在噪声图像（促销遮罩/背景杂乱）场景下自适应调权，实现鲁棒的电商 I2I 多模态检索嵌入。
💡 创新性分析：首次将商品文本元数据作为 Q-Former 的硬性语义约束，明确区分噪声视觉和真实商品特征；双门控调制在噪声下降低探索流、提升文本引导流，解决了工业 I2I 检索中图像噪声这一长期痛点。与 MLP Connector、标准 Q-Former 相比，无需端到端 MLLM 推理，轻量适合工业部署。评分：81/100
📊 关键指标：大规模工业电商全量检索池 Hit Rate@100 (H@100) 平均提升 +6.04%，显著优于 connector baselines 及端到端 MLLM 基线。

---

📄 标题：Valley3: Scaling Omni Foundation Models for E-commerce
👥 作者：Zeyu Chen, Guanghao Zhou, Qixiang Yin 等 (阿里巴巴集团, arXiv 2605.01278, 2026-05-02)
🔗 链接：https://arxiv.org/abs/2605.01278
📝 方法概述：Valley3 是首个覆盖文本/图像/视频/音频的全模态电商专用 MLLM，通过四阶段继续预训练管道（音频理解→跨模态指令→电商领域知识→长上下文推理）逐步赋予 LLM 多模态理解与电商专业能力，重点突破原生多语言音频能力（支持直播带货语音）和内容违规检测（涉黄/盗版/性暗示）。
💡 创新性分析：首个四模态（T+I+V+A）电商专用 MLLM；四阶段渐进训练管道解决了领域知识注入与全模态能力获取的耦合难题；原生多语言音频填补了现有电商模型的音频盲区。相比 Valley2 等前作大幅扩展模态覆盖。评分：80/100
📊 关键指标：自建全模态电商基准（6 类任务）综合 SOTA，在违规检测子任务（涉黄/盗版/性暗示）上显著优于通用 VLM（GPT-4V、Qwen-VL），同时在通用基准（MMBench）保持竞争力。

---

📄 标题：CS-VAR: Deja Vu in Plots — Cross-Session Evidence with RAG-LLMs for Live Streaming Risk Assessment
👥 作者：Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He (中科院计算所, arXiv 2601.16027, 2026-01-22)
🔗 链接：https://arxiv.org/abs/2601.16027
📝 方法概述：CS-VAR 针对直播平台跨会话复现的诈骗/协同恶意行为，设计"双模型+RAG"架构：LLM 对检索到的历史风险会话证据进行跨会话模式推理，并将这种全局洞察通过蒸馏迁移到轻量级实时 Student 模型，使实时检测器具备"历史记忆"而无需在线调用 LLM。
💡 创新性分析：将跨会话 RAG 与 LLM-to-Student 知识蒸馏结合，赋予轻量实时风控模型识别历史重复风险剧情的能力——这是现有单会话风控模型完全不具备的。首次将 RAG 引入直播实时风险检测领域。评分：76/100
📊 关键指标：真实直播平台数据集上 F1 和 AUC 显著优于 RNN/GNN 单会话基线（具体数值待直接访问论文）。

---

📄 标题：PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media
👥 作者：Zoher Kachwala, Bao Tran Truong, Rasika Muralidharan, Haewoon Kwak, Jisun An, Filippo Menczer (印第安纳大学/德累斯顿工业大学, arXiv 2605.17187, 2026-05-16)
🔗 链接：https://arxiv.org/abs/2605.17187
📝 方法概述：构建首个大规模多元化社区内容审核基准 PluRule，覆盖 1,989 个 Reddit 社区/2,885 条社区规则/9 种语言，包含 13,371 条人工验证违规样本。将审核任务形式化为多选题式规则匹配（识别评论违反哪条规则），系统评测 SOTA VLM 在社区异质性规则下的推理能力。
💡 创新性分析：首个聚焦多元化社区规则异质性的多模态多语言审核基准；揭示即使 GPT-5.2 在此类细粒度规则推理任务上也几乎与随机基线持平，系统诊断了当前 VLM 在复杂治理推理上的深层缺陷。评分：65/100
📊 关键指标：GPT-5.2 在 PluRule 上仅略好于随机基线（~1/N 规则）；通用规则（文明/自我推广）检出率高于长尾社区特定规则。

---

📄 标题：GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion
👥 作者：Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He (新加坡国立大学, arXiv 2605.00670, 2026-05-01)
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：针对真实推荐系统中普遍存在的模态缺失问题，GRE-MC 通过模态感知子图检索（从全局商品交互图中找到语义相近商品）为缺失模态提供上下文，再用图 Transformer 全局注意力联合编码实现高质量补全，并通过可学习稀疏路由码本正则化嵌入鲁棒性。
💡 创新性分析：首次将子图检索引入多模态推荐缺失模态恢复，相比传统零填充/均值填充方法，充分利用了图结构中的跨商品语义关联。评分：62/100
📊 关键指标：Amazon Baby/Sports/Clothing 多数据集缺失模态场景下 Recall@20 和 NDCG@20 持续优于 FREEDOM、BM3 等 SOTA 方法。

---

📄 标题：SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking
👥 作者：Ruochen Yang, Yueyang Liu 等 17 人 (快手科技, arXiv 2602.09401, 2026-02-10)
🔗 链接：https://arxiv.org/abs/2602.09401
📝 方法概述：SARM 将每个直播间的语义表示设计为一组可学习语义锚点 token，与排序特征端到端联合优化——使 LLM 文本描述直接受到排序目标梯度驱动，兼顾动态内容语义建模与实时服务效率（非对称部署）。
💡 创新性分析：语义锚点 token 与排序 loss 端到端联合优化是核心创新，解决了密集嵌入与排序目标弱对齐的根本问题；非对称部署策略兼顾 LLM 语义质量与实时推理效率。已在快手 4 亿日活规模 A/B 测试验证。评分：85/100（Feb 2026，仅参考）
📊 关键指标：快手 4 亿+ 用户规模线上 A/B 测试中用户在线时长与互动指标显著提升；离线评测一致改善。
