📄 标题：Beyond Relevance-Centric Retrieval: Rubric-Oriented Document Set Selection and Ranking
👥 作者：Kailin Jiang, Lei Liu, Jian Xi, Hui Xu, Junlin Liu, Baochen Fu, Shaoqing Ren, Bin Li, Vichwang, Yu Lu, Haibo Shi
🔗 链接：https://arxiv.org/abs/2607.19747
📝 方法概述：论文把检索目标从“单文档相关性”升级为“文档集合质量”，用 2.8 万条 query-specific rubrics 描述互补性、冗余、冲突和真实性，并进一步提出 RUBRIC4SETWISE 直接按 rubric 做 setwise 选择与排序。
💡 创新性分析：核心创新是重新定义 RAG 证据检索的优化目标，强调证据集合的覆盖度、互补性和真实性；对电商内容治理中的证据核查、冲突内容识别和多样性控制很有落地价值。
📊 关键指标：rubric 覆盖得分与下游生成质量 Pearson r=0.92；短文本多跳 QA 上平均分 27.71（EM 26.10 / F1 29.32），平均仅需 2.66 篇文档；长文本场景 LLM-judge 70.57，优于次优方法且使用更少文档和轮次。

📄 标题：UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction
👥 作者：Honghao Li, Xianquan Wang, Kangyi Lin, Zibin Zhang, Yiwen Zhang, Yi Zhang
🔗 链接：https://arxiv.org/abs/2607.19987
📝 方法概述：UniRank 建立了工业级排序统一基准，覆盖 15 种模型和 5 个亿级公开数据集，统一行为序列建模与特征交互训练范式，并提供高性能 PyTorch 工具包支撑长序列排序研究。
💡 创新性分析：创新点在于把原本分散的长序列排序研究统一到可复现 benchmark 中，对电商搜索/推荐/内容分发的模型选型、扩展和线上复验都极具指导价值。
📊 关键指标：覆盖 QK-Video（4.9 亿）、KuaiRand（3.2 亿）、TAAC-25（7.5 亿）等数据集；KuaiRand 上 EST 的 Long View AUC 达到 0.8157；工具包在 4 卡 H20 上实现 14.24 倍吞吐加速。

📄 标题：Personalized Recommendation Tool Learning via Autonomous Language Agents
👥 作者：Mingdai Yang, Zhiwei Liu, Weizhi Zhang, Yibo Wang, Hao Peng, Philip Yu
🔗 链接：https://arxiv.org/abs/2607.19739
📝 方法概述：PRTA 将传统推荐模型视为可调用工具，由 LLM 代理通过 Personalized Tool Memory 和记忆反思机制，在用户级别学习调用何种推荐器以及如何聚合多模型结果。
💡 创新性分析：把推荐问题改造成 Agent 的“工具学习”问题，兼顾了 LLM 的语义理解能力与传统推荐器处理全量候选的工程优势，适合电商多内容形态与多用户偏好的调度场景。
📊 关键指标：Amazon 上 Recall@10 从 0.0625 提升到 0.1000（约 +60%）；Goodreads 上 Recall@10 从 0.1000 提升到 0.1688（约 +68.8%）。

📄 标题：Detect Early, Escalate Rarely: Anytime Detection of AI-Generated Video from the Compressed Bitstream
👥 作者：Mert Onur Cakiroglu, Mehmet Dalkilic, Hasan Kurban
🔗 链接：https://arxiv.org/abs/2607.19476
📝 方法概述：论文把 AI 生成视频检测改造成“码流轻特征预筛 + 少量难例升级”的流式治理框架，通过 anytime-valid 阈值在视频尚未完整解码时就进行风险判断。
💡 创新性分析：创新在于把检测入口前移到压缩码流层，大幅降低海量视频治理成本；对达人内容、短视频和直播切片的 AIGC 检测非常实用。
📊 关键指标：公开网页信息显示其 CPU 计算量比像素级 CNN 低 5 个数量级；GenVidBench 全长 AUC 为 0.64；仅升级约 15% 片段时准确率由 0.75 提升到 0.78，总算力仍约低 7 倍。

📄 标题：Stochastic Primal-Dual Decoding for Multiobjective Generative Recommender Systems
👥 作者：Denis Beslic, Melissa Yalla, Alice Y Wang, Mounia Lalmas
🔗 链接：https://arxiv.org/abs/2607.19357
📝 方法概述：论文把生成式推荐重写为带约束的多目标序列生成，通过随机原对偶解码在生成时动态调整 logits，兼顾相关性、公平性、商业目标和新内容曝光，无需额外重排。
💡 创新性分析：将多目标约束直接塞进 decoding，而不是改训练损失或事后重排，特别适合电商内容分发中对转化、多样性和达人曝光公平的在线权衡。
📊 关键指标：公开网页信息显示 Spotify 在线 A/B 中辅助目标提升约 +1.8%，同时用户满意度相关指标基本零损失；离线 trade-off 表现优于传统启发式和 reranking 方案。

📄 标题：Silent Failures in Multimodal Agentic Search: A Diagnostic Taxonomy and Cross-Judge Evaluation
👥 作者：Zhengxian Wu, Junjie Gao, Kai Yang
🔗 链接：https://arxiv.org/abs/2607.19793
📝 方法概述：论文定义了多模态 Agent Search 中的六类“静默失败”，并通过人工、LLM、VLM 的 cross-judge 评估框架，诊断最终答案正确但证据轨迹错误的风险。
💡 创新性分析：它把多模态搜索 Agent 的“过程正确性”独立出来评估，对电商拍照搜同款、AI 导购问答和多模态证据检索系统的可信度治理有很大参考价值。
📊 关键指标：公开网页信息表明论文系统性揭示了静默失败会随模型能力提升而迁移，说明仅看最终答案准确率不足以度量真实可靠性。
