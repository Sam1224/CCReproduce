# Feishu Cards — 2026-06-17 Daily Paper Inspection
# Papers with score ≥ 40 (sorted by score descending)

---

📄 标题：Atomic Intent Reasoning (AIR): Bringing LLM Semantics to Industrial Cross-Domain Recommendations
👥 作者：Zhuohang Jiang, Yuxin Chen, Shijie Wang, Haohao Qu, Jindong Zhou, Wenqi Fan, Qing Li, Dongxu Liang, Jun Wang（香港理工大学 + 快手科技）
🔗 链接：https://arxiv.org/abs/2606.10357
📝 方法概述：AIR 将 LLM 推理迁移至离线阶段，将用户行为分解为原子意图（Atomic Intent）向量并建立索引；在线阶段仅做轻量 ANN 检索 + 线性组合，无需在线 LLM 调用，实现 ~400× 推理加速，将内容侧用户意图桥接至电商转化侧排序特征。
💡 创新性分析：首个将"离线 LLM 推理 + 在线原子意图组合"范式系统化应用于工业级跨域推荐，在保证语义质量的前提下实现 400× 延迟压缩，与传统 MMOE/PLE 跨域方法相比引入了语义层面的跨域桥接。
📊 关键指标：快手电商线上 A/B 实验 GMV +3.446%；多项核心业务指标显著提升；推理加速约 400×（vs 在线 LLM 基线）

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng（阿里巴巴国际数字商业集团）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：QueryAgent-R1 将查询推荐定义为 Agent 任务，使用链式检索优化（Chain-of-Retrieval）——Agent 生成候选查询后主动调用商品检索器，基于检索结果和用户历史偏好进行端到端打分与 R1-style 强化学习训练，实现查询生成与商品检索的闭环对齐，突破了传统方案中查询生成与检索结果对齐不足的困境。
💡 创新性分析：首次将检索器行为纳入查询推荐 Agent 的决策回路，R1-style RL 直接以 CVR 信号优化 Agent，实现 CTR+CVR 联合提升而非传统的单一查询相关性优化。
📊 关键指标：阿里国际线上 A/B 实验 Query CTR +2.9%，guided CVR +3.1%；离线多数据集一致提升

---

📄 标题：OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation
👥 作者：Jiakai Tang, Sunhao Dai, Kun Wang et al.（中国人民大学 + Shopee + 南洋理工大学）
🔗 链接：（工业论文，RecSys/KDD 2026）
📝 方法概述：OneRank 提出 Transformer-native 多任务统一排序框架，所有任务特征与交互在单一 Transformer 主干中处理，通过任务私有通道（Task-Private Channels）和梯度脱钩（Gradient Decoupling）避免任务干扰，同时保持参数共享优势。在 Shopee 主排序上完成大规模线上验证。
💡 创新性分析：首个将 Transformer 自注意力深度集成到工业级多任务排序的系统，任务私有通道设计在 Transformer 层面隔离任务梯度，超越传统 MMOE/PLE 的浅层门控机制。
📊 关键指标：Shopee 主排序线上 A/B 实验：GMV 及多项指标显著提升；推理效率维持工业可部署标准

---

📄 标题：LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams
👥 作者：Zhenyu Yang, Kairui Zhang, Bing Wang, Shengsheng Qian, Changsheng Xu（中国科学院自动化研究所）
🔗 链接：https://arxiv.org/abs/2606.17798
📝 方法概述：LiveStarPro 针对在线直播理解的三重困境（流式处理、自主判断回复时机、长程记忆）提出 SVeD（单次困惑度验证自动判断回复时机）、SCAM（流式因果注意力掩码增量训练）、TSHM（树形层次化记忆递归压缩历史）三组件协同方案，实现主动式、持续性、层次化记忆的实时直播内容理解。
💡 创新性分析：首次将"主动式回复时机判断"系统性纳入流式 Video-LLM 训练框架，树形记忆支持实际无界视频流的高效检索，直接适用于电商直播违规内容检测与精彩片段识别。
📊 关键指标：vs 先前在线 Video-LLM：语义正确率 +28.9%，时机误差 -18.2%，推理加速 1.58×

---

📄 标题：Harmonizing Semantic and Collaborative in LLMs: Reasoning-based Embedding Generator for Sequential Recommendation
👥 作者：Qidong Liu, Mingyao Huang, Moranxin Wang, Wenxuan Yang, Haiping Zhu（西安交通大学）
🔗 链接：（待确认 arXiv ID）
📝 方法概述：将 LLM 改造为推荐 Embedding 生成器，通过潜在推理（Latent Reasoning）+ 共现奖励强化学习，使 LLM 生成的物品向量同时融合语义与协同过滤信号，离线预计算无在线推理开销，可与任意推荐 Backbone 即插即用。
💡 创新性分析：RL 驱动的 LLM 语义-协同信号融合是新颖组合，长尾物品推荐显著改善，跨 backbone 与跨 LLM 泛化稳定。
📊 关键指标：多数据集 NDCG/HR 持续优于 SASRec、GRU4Rec 等基线；长尾物品改善尤为显著

---

📄 标题：MLT-Dedup: Efficient Large-Scale Online Video Deduplication via Multi-Level Representations and Spatial-Temporal Matching
👥 作者：David Yuchen Wang, Haoying Li, Hailun Xu, Wei Chee Yew et al.（工业界，KDD-2026 ADS track）
🔗 链接：https://arxiv.org/abs/2606.12215
📝 方法概述：MLT-Dedup 使用多级视频编码器（稀疏 clip 级 + 精细 frame 级）双路 Embedding，稀疏路快速 ANN 候选召回，精细路精准匹配，配合时空动态规划对齐模块处理片段重排，构建可在线流式处理大规模视频平台的去重系统。
💡 创新性分析：两级 Embedding 兼顾效率与精度，时空匹配模块专门处理近重复视频的片段级重排问题，KDD-2026 ADS 赛道背书工业规模验证。
📊 关键指标：KDD-2026 ADS 验证：生产级精度/召回；在线流式架构满足大规模平台实时需求

---

📄 标题：Detecting AI-Generated Content on Social Media with Multi-modal Language Models
👥 作者：Chenyang Yang, Shen Yan, Yibo Yang et al.（卡内基梅隆大学 + Meta）
🔗 链接：https://arxiv.org/abs/2606.11200
📝 方法概述：提出一套端到端多模态 AIGC 检测流水线：跨平台持续策划多样化社交媒体数据（图、视频、文本），训练紧凑型视觉-语言模型（compact VLM）同时输出检测判断与可解释文字说明，已在 Meta 平台部署，对帖子推荐产生正向用户互动影响。
💡 创新性分析：持续数据策划机制（Continuous Curation）能跟踪新生成模型，解决静态训练集失效问题；多模态+可解释输出对内容治理审核员有直接价值；Meta 规模部署验证工业可用性。
📊 关键指标：公开 AIGC 基准达 SOTA；内部多平台数据集鲁棒检测；Meta 部署后用户互动正向提升

---

📄 标题：One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets
👥 作者：Woosung Kang, Jiwon Jeong, Jonghyeok Shin, Jeongwhan Choi, Noseong Park（KAIST）
🔗 链接：（待确认 arXiv ID）
📝 方法概述：首次将 PFN/后验预测范式引入序列推荐：在合成先验数据上预训练通用序列推荐模型，推理时单次前向传播即可适配任意新数据集（update-free），无需梯度更新，实现新类目零样本泛化。
💡 创新性分析：PFN + 序列推荐首次结合，update-free 单次适配效率极高（1 分钟 vs 15 小时），对电商新类目冷启动有直接价值。
📊 关键指标：适配时间：1 分钟 vs 15 小时微调基线；未见类目零样本推荐 Recall 竞争力强

---

📄 标题：On the Memorization Behavior of LLMs in Generative Recommendation
👥 作者：Sunwoo Kim, Sunkyung Lee, Clark Mingxuan Ju et al.（KAIST + Sungkyunkwan + Snap）
🔗 链接：（待确认 arXiv ID）
📝 方法概述：首次系统量化 LLM 生成式推荐中的"一跳记忆"现象，发现高 Recall 部分来自训练集记忆而非泛化；提出 IIRG 多任务训练策略，增加协同邻居生成和语义邻居生成任务，改善长尾用户推荐质量。
💡 创新性分析：诊断角度清晰，IIRG 即插即用；17 基线对比实验扎实；Snap 工业背景提升应用价值。
📊 关键指标：vs 17 个基线：SID/TID 双指标改善；长尾用户 Recall 显著提升

---

📄 标题：Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer
👥 作者：Wujian Peng, Lingchen Meng et al.（复旦大学 + Qwen Team, Alibaba）
🔗 链接：（ICML 2026）
📝 方法概述：提出共享上下文-视觉 Tokenizer，使用单一离散 Tokenizer 统一处理图像生成与理解，结合二值量化 + 并行比特自回归，在生成、编辑、文本渲染多项任务上超越 GPT-4o，同时保持高效率。
💡 创新性分析：共享 Tokenizer 消除"理解 vs 生成"的模态鸿沟，并行比特自回归加速生成，多任务统一是 ICML 2026 重要成果。
📊 关键指标：生成/编辑/文本渲染多项指标超越 GPT-4o；ICML 2026 接受

---

📄 标题：Do Generative Recommenders Deepen the Information Cocoon?
👥 作者：Jiyuan Yang, Gengxin Sun, Mengqi Zhang et al.（山东大学 + 中国人民大学）
🔗 链接：（待确认 arXiv ID）
📝 方法概述：构建 LLM-agent 闭环仿真系统，用 LLM 驱动虚拟用户在生成式 vs 传统推荐器下持续交互，提出代码空间结构茧房指标量化内容同质化程度，首次量化比较 GenRec 与传统推荐的信息茧房效应。
💡 创新性分析：首个 LLM-agent 闭环仿真对比研究，原创茧房量化指标，对内容多样性治理有重要政策参考价值。
📊 关键指标：多推荐器多场景仿真对比；内容多样性、长尾曝光、头部集中度三维指标

---

📄 标题：Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations
👥 作者：Nimesh Sinha, Raghav Saboo, Martin Wang, Sudeep Das（DoorDash Inc.）
🔗 链接：https://arxiv.org/abs/2606.06779
📝 方法概述：针对多业态平台（餐厅/商超等）行为孤岛问题，利用分层 RAG 从数据丰富业态（餐厅）提取多级分类特征，通过 LLM 生成式推断将其注入 MTL 主排序模型，实现跨业态知识迁移。
💡 创新性分析：分层 RAG + LLM 特征生成 + MTL 集成的多业态推荐工程方案，在 DoorDash 生产中验证，对多业务线电商平台有参考价值。
📊 关键指标：DoorDash 线上 A/B 实验指标提升（具体数值未公开）

---

📄 标题：MODE-RAG: Manifold Outlier Diagnosis and Energy-based RAG Evaluation
👥 作者：Zehang Wei, Jiaxin Dai, Jiamin Yan, Xiang Xiang（华中科技大学）
🔗 链接：（待确认 arXiv ID）
📝 方法概述：从流形视角诊断 RAG 检索空间中的离群点，利用变分自由能构建能量基门控机制动态过滤低质量检索文档，配合多智能体架构实现多模态内容质量治理。
💡 创新性分析：流形离群点 + 变分自由能门控组合理论新颖，对多模态 RAG 幻觉抑制有新视角。
📊 关键指标：自建基准上幻觉率下降；多模态内容质量评估指标改善

---

📄 标题：Leveraging Code-Mixed Product Metadata for Personalized Recommendation on Daraz Bangladesh
👥 作者：KM Fahim A Bari, Muhammad Abdullah Adnan, Nafis Sadeq（东西大学 + BUET）
🔗 链接：（待确认 arXiv ID）
📝 方法概述：首个孟加拉电商平台 Daraz 的可复现推荐基准，量化混合语言（Banglish）商品元数据对推荐效果的影响，结合多语言预训练模型改善低资源语言场景推荐。
💡 创新性分析：低资源语言电商推荐首个基准，量化词汇碎片化对推荐的损害，对全球化电商有参考价值。
📊 关键指标：Daraz Bangladesh BanglishRev 数据集 Top-N 推荐指标（NDCG/HR）；混合语言预训练 vs 英语基线对比

---

📄 标题：When More Documents Hurt RAG: Mitigating Vector Search Dilution
👥 作者：（待确认）
🔗 链接：https://arxiv.org/abs/2606.11350
📝 方法概述：发现大规模文档集中的向量搜索稀释现象，提出领域作用域（Domain-Scoped）先过滤再密集检索的方案，显著提升大规模异构文档集上的 RAG 精度，模型无关。
💡 创新性分析：简洁有效的 RAG 扩展方案，模型无关设计有广泛适用性。
📊 关键指标：多大规模 RAG 基准上准确率显著提升；2× FLOPS 减少（部分实验）
