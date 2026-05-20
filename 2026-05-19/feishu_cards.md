# Feishu Cards — 2026-05-19

---

📄 标题：When Rules Fall Short: Agent-Driven Discovery of Emerging Content Issues in Short Video Platforms
👥 作者：Chenghui Yu, Zixuan Wang, Hongwei Wang, Bingfeng Deng, Junwen Chen, Zhuolin Hao, Hongyu Xiong（TikTok Inc.）
🔗 链接：https://arxiv.org/abs/2601.11634
📝 方法概述：提出多模态 LLM Agent 自动发现短视频平台新兴内容问题的系统。Agent 先从大量视频中召回潜在违规内容，再通过两阶段聚类区分"已知问题变体"与"全新子问题"，最终自动生成更新的标注策略，形成治理闭环。
💡 创新性分析：首次系统化解决新兴内容问题的自动发现问题；两阶段聚类方法可有效区分变体与新问题；LLM Agent 的多模态推理能力超越规则/关键词方法；已在 TikTok 平台实际部署，具备工业验证。
📊 关键指标：F1 提升 >20%（vs 人工发现）；违规视频曝光减少约 15%；策略迭代周期大幅缩短（MicroLens/Short-Video 平台真实指标）

---

📄 标题：E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs
👥 作者：Xianjie Liu, Yiman Hu, Liang Wu, Ping Hu, Yixiong Zou, Jian Xu, Bo Zheng（阿里巴巴/淘宝）
🔗 链接：https://arxiv.org/abs/2602.08355
📝 方法概述：提出首个电商短视频理解基准，含淘宝真实视频 3961 条和 19785 个开放式 Q&A 对。通过多智能体标注系统规模化生成高质量训练数据，并构建多模态信息密度量化框架证明电商视频的高难度特性。基于 RL 微调的 E-VAds-R1 模型在商业意图推理上实现跨越式提升。
💡 创新性分析：首个电商短视频专属 MLLM 评测基准；信息密度量化框架客观揭示电商视频复杂性；多智能体标注解决了大规模标注成本问题；少量样本 RL 微调实现 109.2% 提升，效率极高。
📊 关键指标：商业意图推理 +109.2%（E-VAds-R1 vs 基础 MLLM，仅用数百条训练样本）；电商视频信息密度显著高于通用数据集。

---

📄 标题：TIGER-FG: Text-Guided Implicit Fine-Grained Grounding for E-commerce Retrieval
👥 作者：（arXiv:2605.18434，2026年5月提交）
🔗 链接：https://arxiv.org/abs/2605.18434
📝 方法概述：针对电商图搜中的 IMMR 任务（局部目标图检索完整商品页），提出文本引导的隐式细粒度 Grounding 方案。用商品结构化文本作语义引导，对全图 patch 做 Cross-Attention 聚合，生成目标感知 embedding，配合双蒸馏策略（空间关系蒸馏+相似度分布蒸馏），实现无需显式检测框的端到端检索。
💡 创新性分析：IMMR 任务定义贴合真实电商图搜场景；文本语义引导视觉注意力实现隐式 Grounding，避免显式检测开销；双蒸馏在无框标注时提供有效监督信号；模型在线服务友好。
📊 关键指标：电商 IMMR 基准上 Recall@K 一致超越基线；双蒸馏 ablation 验证各组件有效性；训练时间/GPU 内存大幅节省。

---

📄 标题：CVA: Compressed Video Aggregator for Efficient Micro-Video Recommendation
👥 作者：Yang Xiao, Huiyuan Chen, Kaiyuan Deng, Chao Jiang, Zinan Ling, Ruimeng Ye, Xiaolong Ma, Bo Hui
🔗 链接：https://arxiv.org/abs/2605.08810
📝 方法概述：针对微视频推荐中 VFM 集成计算成本高、帧冗余严重的问题，提出 CVA 轻量模块：冻结 VFM 参数、用 CLIP+标题做语义关键帧重采样、通过隐式推理聚合多帧 embeddings，生成紧凑视频表示，作为即插即用模块嵌入任意推荐 backbone。
💡 创新性分析：语义关键帧重采样解决帧冗余问题，本身具有独立价值；解耦策略（冻结 VFM）数量级降低训练成本；模块化设计适配多种推荐系统；首次系统分析了基准数据集帧采样策略的缺陷。
📊 关键指标：MicroLens 和 Short-Video 两个基准上 Recall/NDCG 持续提升；训练时间和 GPU 内存降低数量级。

---

📄 标题：AuDisAgent: Training-Free Multimodal Controversy Detection Multi-Agent Framework
👥 作者：Zihan Ding（四川大学）, Ziyuan Yang（南洋理工大学）, Yi Zhang（四川大学）
🔗 链接：https://arxiv.org/abs/2605.02939
📝 方法概述：将多模态争议性内容检测（MCD）从静态特征分类重新定义为受众动态传播过程。三类筛选 Agent（Video Agent、Comment Agent、Interaction Agent）分别从视觉、文本、跨模态交互角度模拟不同受众群体的评价，通过 LLM 推理综合多方视角，免训练实现可解释争议检测。
💡 创新性分析：受众传播视角突破了 MCD 的静态建模范式；免训练架构部署成本低且随 LLM 进步自动受益；三 Agent 专业化分工提供检测结果可解释性；对新型争议话题有零样本泛化能力。
📊 关键指标：在社交视频 MCD 基准上超越监督学习基线（具体数值待原文发布）；提供定性可解释性优于端到端方法。

---

📄 标题：Thinking Broad, Acting Fast: Latent Reasoning Distillation for E-Commerce Relevance
👥 作者：Baopu Qiu, Hao Chen, Yuanrong Wu, Changtong Zan, Chao Wei, Weiru Zhang, Xiaoyi Zeng
🔗 链接：https://arxiv.org/abs/2601.21611
📝 方法概述：针对电商搜索相关性判断，提出多视角 CoT 覆盖相关性多维本质（用户意图/属性匹配/业务规则），并将多视角推理蒸馏到紧凑学生模型的隐式空间，实现"推理能力强+延迟低"的双重目标。发表于 WWW 2026。
💡 创新性分析：多维相关性建模比单视角 CoT 更贴近业务现实；隐式推理蒸馏是 KD 领域的方法创新；在线 A/B 实验验证了实际业务价值；WWW 2026 顶会认可。
📊 关键指标：AUC/F1 超越单视角 CoT 基线；在线 A/B 实验显示 GMV/CTR 正向提升；推理延迟接近轻量模型。

---

📄 标题：MAP-V: Transparent Recommendation Filtering via Multimodal Multi-Agent Collaboration
👥 作者：Chi Zhang, Zhipeng Xu, Jiahao Liu, Dongsheng Li, Hansu Gu, Peng Zhang, Ning Gu, Tun Lu（复旦大学、微软亚洲研究院）
🔗 链接：https://arxiv.org/abs/2604.17459
📝 方法概述：解决推荐过滤层缺乏多模态感知和过度关联误判的问题。MAP-V 通过可编辑偏好图+四 Agent 协作（Intent Parser、Judge、Dispute、RAH）实现精准意图对齐，端云协同架构平衡延迟与准确率。
💡 创新性分析：可编辑偏好图使过滤决策透明可追溯；Dispute Agent 专门处理边界案例减少误判；端云协同架构具有工程实用价值；微软亚洲研究院参与保证了系统级考量。
📊 关键指标：Precision/Recall 超过 LLM-only 基线；过度关联误判率显著降低；多模态不当内容检测优于纯文本方法。

---

📄 标题：GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion
👥 作者：Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He（新加坡国立大学）
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：针对真实多模态推荐数据中的模态缺失问题，GRE-MC 通过模态感知子图检索从整体图中获取语义相关上下文，图 Transformer 联合编码补全缺失特征，可学习稀疏路由码本正则化提升鲁棒性。
💡 创新性分析：子图级上下文用于模态补全超越了节点级插值；图 Transformer 全局注意力兼顾局部结构和全局语义；码本正则化增强了对噪声的鲁棒性；NUS 顶级团队出品。
📊 关键指标：Amazon Baby/Sports/Clothing 三个基准上 Recall@10、NDCG@10 持续超越 SOTA 方法。

---

📄 标题：Who Decides What Is Harmful? Content Moderation via Multi-Agent Personalised Inference
👥 作者：Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：提出 LLM 多智能体个性化内容审核框架，包含 Expert Agents（领域专家）、Manager Agent（协调者）、Ghost Profile Agent（用户视角模拟），基于用户独特敏感性档案过滤内容，替代一刀切的中心化审核规则。
💡 创新性分析：Ghost Profile Agent 模拟用户视角是个性化审核的新思路；多 Expert Agent 协作提升决策覆盖范围；32% 准确率提升证明个性化的价值；但电商/达人场景的直接适用性有限。
📊 关键指标：准确率相比非个性化基线提升 32%；用户敏感度对齐度显著改善。
