# Feishu Cards — 2026-06-09 AI 论文巡检（补充场次）

> 本文件包含本次巡检中评分 >= 40 的所有新论文（未在前次巡检中收录）的飞书推送卡片。

---

📄 标题：FLUID — 工业级直播推荐的多模态语义编码
👥 作者：Xinhang Yuan, Zexi Huang, Anjia Cao, Xudong Lu, et al.
🔗 链接：https://arxiv.org/abs/2605.21832
📝 方法概述：FLUID 是首个在生产级直播排序模型中完全退役候选侧 item ID 的工业框架。核心在于跨域多模态编码器 LUCID（联合训练短视频+直播间），将多模态内容映射为离散层次语义编码，完全替代 ID embedding，并通过阶段预热保证在线增量训练的稳定性。
💡 创新性分析：首次在工业规模验证"无 ID 推荐"在直播场景的可行性，解决了直播间永久冷启动的核心痛点；LUCID 跨域联合训练设计在短视频与直播之间建立语义桥接，而非两套独立编码器。
📊 关键指标：工业级在线 A/B 测试，点击率与用户时长均显著提升；跨域新直播间 Recall@20 大幅优于 ID-based baseline。总分 88/100。

---

📄 标题：CQ-SID+EG-GRPO — 电商搜索高效生成式召回
👥 作者：Jianbo Zhu et al. (8 authors)
🔗 链接：https://arxiv.org/abs/2605.14434
📝 方法概述：针对工业电商搜索中生成式召回（GR）落地的三大瓶颈（海量目录、延迟要求、目标对齐），提出 CQ-SID（类目+查询双约束语义聚类 ID）压缩 beam search 空间，以及 EG-GRPO（专家引导的 GRPO 强化学习）对齐下游排序目标，将 GR 定位为召回阶段的高效补充。
💡 创新性分析：双约束 SID 设计让语义 ID 既考虑类目层级又考虑查询意图，专家引导解决了纯在线奖励稀疏问题；务实地将 GR 定位为召回补充而非全面替代，降低了风险。
📊 关键指标：工业电商搜索 Recall@50 和 NDCG 显著提升，beam search 复杂度大幅降低。总分 84/100。

---

📄 标题：QueryAgent-R1 — 基于 Chain-of-Retrieval 的电商查询推荐
👥 作者：Ant Group / Alibaba 研究团队
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：针对电商查询推荐中"查询 CTR 高但商品转化率低"的脱节问题，提出记忆增强 Agentic 框架 QueryAgent-R1，通过 chain-of-retrieval 在查询生成过程中实时检索实际商品库存，并以一致性奖励联合优化查询相关性和下游商品参与度（GRPO 策略优化）。
💡 创新性分析：将 chain-of-thought 思路延伸到"检索感知生成"，强制要求生成查询必须可达高质量商品，从根本上解决 CTR-CVR 脱节；记忆模块缓存历史检索结果降低推理开销。
📊 关键指标：查询推荐 CTR 和 CVR 同步提升，在线 A/B 测试综合参与度有效改善。总分 74/100。

---

📄 标题：TGQ-Former — 文本引导视觉表征的多模态电商推荐
👥 作者：电商研究团队
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：提出文本引导 Q-Former（TGQ-Former），以商品结构化元数据（标题、属性）为语义 Query，通过跨注意力从商品图像 token 中精准提取与商品描述对齐的视觉特征，同时保留互补视觉证据，降低背景噪声干扰。
💡 创新性分析：区别于直接 CLIP 编码，TGQ-Former 在特征提取阶段就实现跨模态对齐，而非依赖后期融合；以业务现有元数据为 query 是零额外标注成本的工程友好设计。
📊 关键指标：电商多模态推荐 HR@K 和 NDCG@K 改善，噪声鲁棒性显著优于 CLIP baseline。总分 72/100。

---

📄 标题：Meta MLLM Framework — 工业级推荐系统的多模态理解通用框架
👥 作者：Meta Platforms 研究团队
🔗 链接：https://arxiv.org/abs/2605.09338
📝 方法概述：Meta Platforms 提出三部架构：离线 MLLM（LLaMA2-based）生成多媒体内容描述 caption → tokenize 为离散类目特征 → 无缝注入现有推荐管道，在不增加在线推理延迟的前提下将 MLLM 的语义理解能力引入工业推荐。
💡 创新性分析：工程框架贡献为主，通过离线-在线解耦解决 MLLM 部署延迟问题；离散化设计使语义特征与现有特征工程体系兼容，落地门槛低。
📊 关键指标：Meta 内部工业推荐系统 CTR 和用户参与度提升（具体数值不公开）。总分 72/100。

---

📄 标题：SearchSwarm — 长程研究的 Agentic LLM 委派智能
👥 作者：Pu Ning, Quan Chen, Jun Zhou et al. (Tsinghua, PKU, Ant Group, RUC)
🔗 链接：https://arxiv.org/abs/2606.09730
📝 方法概述：针对 LLM 上下文窗口限制导致长程任务处理困难的问题，提出委派智能（Delegation Intelligence）：主 Agent 将任务分解并委派给子 Agent，子 Agent 只返回摘要，保护主 Agent 上下文预算；构建专用训练数据微调开源模型 SearchSwarm-30B-A3B。
💡 创新性分析：将"委派"操作化为可训练能力，填补开源社区委派智能空白；BrowseComp 上同规模最优结果；发表于 ICML 2026 SCALE 研讨会（口头报告）。
📊 关键指标：BrowseComp 68.1 / BrowseComp-ZH 73.3（30B-A3B 规模 SOTA）。总分 68/100。

---

📄 标题：Unintended Consequences — 推荐系统干预的意外后果田野实验
👥 作者：Shilei Luo, Song Yao, Dennis J. Zhang
🔗 链接：https://arxiv.org/abs/2606.08265
📝 方法概述：在短视频平台上进行大规模随机对照实验：平台发起"睡眠提醒"干预（减少深夜使用），结果深夜参与度反而上升 14.75%，整体使用量增加 2.18%，且效果持续数周。机制是"强制探索效应"——干预触发算法策略更新，形成自我强化循环。
💡 创新性分析：首次通过工业 RCT 量化"治理干预→算法学习→意外后果"的完整链条，对平台内容治理策略设计有重要警示；揭示静态评估框架在动态推荐系统中的失效边界。
📊 关键指标：深夜参与度 +14.75%（与预期相反），整体平台使用量 +2.18%，持续效果数周。总分 62/100。

---

📄 标题：Memory Beyond Recall — LLM Agent 的双过程认知记忆系统
👥 作者：Tianxiang Fei et al. (Tencent)
🔗 链接：https://arxiv.org/abs/2606.09483
📝 方法概述：受认知科学双过程理论启发，为 LLM Agent 设计双层记忆：System 1（快速 embedding 相似度检索）处理高频模式；System 2（结构化推理）处理罕见重要信息；Agent 通过任务反馈自动更新两类记忆，实现自进化。
💡 创新性分析：将认知科学双过程理论系统性地映射到 LLM Agent 记忆架构，提供比纯 RAG 更细粒度的记忆管理；腾讯工业研究背景保证落地可行性。
📊 关键指标：多步骤推理任务成功率提升，长期任务性能保持更稳定。总分 55/100。
