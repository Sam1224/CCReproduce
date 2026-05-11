# Feishu Cards — 2026-05-10 Daily AI Paper Inspection (BACKFILL)

> 以下为得分 ≥ 40 的论文飞书卡片，按得分从高到低排序。

---

📄 标题：Valley3: Scaling Omni Foundation Models for E-commerce
👥 作者：Zeyu Chen, Guanghao Zhou, Qixiang Yin, Ziwang Zhao, Huanjin Yao, et al.（AIDC-AI / 阿里国际数字商业集团）
🔗 链接：https://arxiv.org/abs/2605.01278
📝 方法概述：Valley3 是首个面向全球化电商场景的 Omni 多模态大语言模型，统一支持文本、图像、视频、音频四模态理解与推理。采用四阶段电商持续预训练（音频理解→跨模态指令遵循→电商领域知识→长上下文推理），并引入可控推理模式（非思考/轻度/重度 CoT）和 Agentic Search 能力，专为短视频电商场景优化原生多语言音频处理。
💡 创新性分析：业界首个端到端 Omni 电商 MLLM，将音频（达人口播/商品讲解）纳入统一预训练；可控推理链适配实时审核与深度研究的不同延迟需求；Agentic Search 解锁多步电商调研任务（比价、竞品分析、合规检查）。相比冻结特征的二阶段范式，Valley3 实现了多模态-推荐目标的端到端对齐。
📊 关键指标：内部 Omni 电商基准（6 任务）+ 开源电商基准 全面超越强 baseline；通用多模态基准保持竞争力。得分：84/100 ✦（进入代码复现）

---

📄 标题：Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection
👥 作者：Tianxiao Li, Zhenglin Huang, Haiquan Wen, et al.（University of Liverpool · CUHK-Shenzhen · NTU · University of Exeter）
🔗 链接：https://arxiv.org/abs/2605.01638
📝 方法概述：Omni-Fake 是首个统一跨四模态（图像/音频/视频/音视频 Talking Head）的社交媒体深度伪造检测基准。包含 Omni-Fake-Set（100 万+样本）和 Omni-Fake-OOD（20 万+分布外样本），支持检测+定位+解释三任务联合评估协议。同时提出 Omni-Fake-R1 检测器（Qwen2.5-Omni-7B + GSPO 强化学习）。
💡 创新性分析：突破现有基准单模态局限；OOD 基准专项测试泛化性；GSPO（组序列策略优化）将 RL 引入多模态深度伪造检测是技术创新。对电商达人/主播内容真实性审核有直接应用价值。
📊 关键指标：Omni-Fake-Set 4 模态检测 AUC 全面优于 single-modal 检测器；Omni-Fake-OOD 分布外泛化 SOTA。得分：76/100

---

📄 标题：AuDisAgent: Training-Free Multimodal Controversy Detection Multi-Agent Framework
👥 作者：Zihan Ding, Ziyuan Yang, Yi Zhang
🔗 链接：https://arxiv.org/abs/2605.02939
📝 方法概述：AuDisAgent 将多模态争议内容检测（MCD）从静态特征学习重定义为"受众传播动力学"问题。三专业筛查 Agent（Video/Comment/Interaction）对视频内容初步评估；当无法达成共识时，Viewing Panel Agent 激活，模拟多元背景受众群体讨论，输出最终判定。全程基于 GLM4-9B 零样本推理，无需训练。
💡 创新性分析：首次将"受众多元性"引入争议检测，突破传统静态分类范式；训练无关意味着可快速迁移至新平台；对短视频电商平台（抖音/快手）内容风控有直接适用性，MMCD 数据集包含 1 万+中文视频。
📊 关键指标：MMCD 数据集（10K+中文视频）：富评论场景 Acc +1.33%、F1 +1.68%；少评论场景 Acc +1.10%、F1 +2.00%（vs. 13 个强 baseline 最优）。得分：72/100

---

📄 标题：The Cost of Context: Mitigating Textual Bias in Multimodal RAG (BAIR)
👥 作者：Hoin Jung, et al.
🔗 链接：https://arxiv.org/abs/2605.05594
📝 方法概述：本文识别 Multimodal RAG 中的"Recorruption"现象——即使完全准确的上下文也会导致 MLLM 放弃原本正确的视觉预测，转而依赖文本。通过注意力矩阵机制诊断，发现"视觉失明"和"位置偏差"两大根因，提出 BAIR（Bottleneck Attention Intervention for Recovery）——无参数推理时框架，恢复视觉显著性并施加位置感知惩罚。
💡 创新性分析："Recorruption"和"成功幻觉（Illusion of Success）"概念对 RAG 评估方法论有深远影响；BAIR 完全免训练，可即插任何 MLLM+RAG 系统，对电商图文商品问答质量有直接提升价值。
📊 关键指标：医疗事实性/社会公平性/地理空间 3 个基准均显著提升多模态基础能力，Recorruption 率大幅降低。得分：66/100

---

📄 标题：MultiSoc-4D: Instruction-Induced Label Collapse in Closed-Set LLM Annotation
👥 作者：（团队待补充）
🔗 链接：https://arxiv.org/abs/2605.06940
📝 方法概述：构建 58K+孟加拉语社交媒体评论的四维度标注基准（类别/情感/仇恨言论/讽刺），使用 ChatGPT/Gemini/Claude/Grok 分别标注。核心发现"指令诱导标签崩溃"：LLM 系统性偏向"回退标签"（Other/Neutral/No），导致79%仇恨内容和75%讽刺内容被漏标，Fleiss' Kappa≈-0.001（近乎随机）。在40+个主流 LLM 上验证该偏差与架构无关。
💡 创新性分析：首次将"标签崩溃"定义为 LLM 自动标注的系统性缺陷，对电商违规内容自动标注系统（如不实宣传/违禁商品检测）具有直接警示意义——基于 LLM 标注数据训练的审核系统可能系统性漏检少数类违规。
📊 关键指标：40+ LLMs 平均：仇恨漏检率79%、讽刺漏检率75%；Sarcasm Kappa≈-0.001（vs. 人工标注 Kappa 显著>0）。得分：65/100

---

📄 标题：Multimodal Data Curation Through Ranked Retrieval (SNS+EEE)
👥 作者：Pratyush Muthukumar, Harshil Kotamreddy, Sarah Amiraslani, Tomo Kanazawa, Ramani Akkati, Shaan Jain, Andrew Mathau
🔗 链接：https://arxiv.org/abs/2605.01163
📝 方法概述：针对多模态嵌入空间中的"模态偏差"（嵌入反映模态而非语义）和"标注噪声"（异构数据集混合后配对质量下降）两大问题，提出 SNS（对称核采样，精炼训练对）和 EEE（专家嵌入引擎，组合互补专家+偏差感知目标）双管齐下的数据策展框架。
💡 创新性分析：训练对级别（SNS）和模型级别（EEE）的双重干预是新颖策略；EEE 可组合现有嵌入专家即插即用；直接适用于电商商品图文配对数据清洗（SKU 图片-标题-描述三模态对齐质量提升）。
📊 关键指标：跨模态检索 Recall@K 多数据集上显著提升；配对一致性分数提升。得分：63/100

---

📄 标题：SIRA: Superintelligent Retrieval Agent
👥 作者：Zeyu Yang, Qi Ma, Jason Chen, Anshumali Shrivastava（Meta AI Research · Rice University）
🔗 链接：https://arxiv.org/abs/2605.06647
📝 方法概述：SIRA 将"超级智能检索"定义为将多轮探索性搜索压缩为单次判别式检索动作的能力。通过语料库侧离线词汇增强（LLM 补充缺失搜索词）和查询侧词汇预测（LLM 预判证据词），结合文档频率过滤工具调用，最终执行一次加权 BM25 检索，完全训练无关。
💡 创新性分析：将 LLM 从"查询重写者"升级为"语料库感知专家"，通过预测"哪些词能区分目标与混淆项"实现单轮超越多轮；开源代码，可直接集成至电商搜索召回层。
📊 关键指标：BEIR 10 个基准 nDCG@10 全面超越密集检索和多轮 agentic 方法；下游 QA 任务显著提升。得分：61/100

---

📄 标题：EPIC: Embedding-based In-Context Prompt Training for LLMs as Text Encoders
👥 作者：Ailiang Lin, Zhuoyun Li, Keyu Mao, Kotaro Funakoshi, Manabu Okumura（Tokyo Institute of Technology）
🔗 链接：https://arxiv.org/abs/2605.01372
📝 方法概述：EPIC 用连续嵌入向量替代离散文本示例作为 In-Context 提示，在保留 ICL 语义对齐效果的同时大幅降低 token 开销。通过对比学习训练 LLM 将演示嵌入解读为语义指导，在 MTEB 基准（公开数据条件下）达到新 SOTA。
💡 创新性分析：离散→连续示例的替换思路在 ICL 嵌入生成中较为新颖；对电商商品语义检索（查询-商品嵌入对齐）有直接应用价值；训练时 token 节省显著。
📊 关键指标：MTEB（公开检索数据训练）综合得分 SOTA，超越 frontier 模型。得分：60/100

---

📄 标题：Topic Is Not Agenda: A Citation-Community Audit of Text Embeddings
👥 作者：Junseon Yoo
🔗 链接：https://arxiv.org/abs/2605.07158
📝 方法概述：通过 358 万篇科学论文的增强引用图（Leiden CPM 两粒度划分），审计 4 个 SOTA 嵌入模型（Gemini/Qwen3-8B/Qwen3-0.6B/SPECTER2）在"研究议程"层级的检索精度：L1 子领域精度 45-52%（尚可），L2 研究议程精度仅 15-21%——10 篇检索结果中 8 篇议程外。
💡 创新性分析：对"余弦相似度=概念相关性"假设的大规模定量证伪，警示 RAG 精度在细粒度领域对齐场景下的系统性失效风险；对电商细粒度品类/违规类型精准匹配有参考价值。
📊 关键指标：4 个 SOTA 模型 L2（研究议程）top-10 同议程率仅 15-21%，即 79-85% 的检索结果"议程外"。得分：54/100

---

📄 标题：EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer
👥 作者：Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng（ICT-CAS / 中科院计算所）
🔗 链接：https://arxiv.org/abs/2605.01732
📝 方法概述：EGAD 通过教师模型输出熵进行 token 级自适应蒸馏：token 级课程学习（低熵→高熵逐步迁移）+自适应温度调节+双分支架构（低熵用 logits 蒸馏，高熵用特征蒸馏），解决传统 KD 对所有 token 等权处理的低效问题。
💡 创新性分析：熵引导的 token 级课程在 LLM 蒸馏中较新颖；双分支架构平衡效率与精度；对压缩内容审核模型（大模型知识迁移至轻量部署模型）有工程价值。
📊 关键指标：相比 token 等权蒸馏 baseline，下游任务性能显著提升；推理效率提升。得分：51/100
