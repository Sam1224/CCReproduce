# Feishu Cards — 2026-06-13 AI Paper Inspection
# 飞书卡片 — 2026-06-13 电商内容生态 & 达人治理 Paper 巡检

---

📄 标题：OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model
👥 作者：Xuxin Zhang, Ben Chen, Yue Lv, et al. (快手 Kuaishou Technology)
🔗 链接：https://arxiv.org/abs/2606.13533
📝 方法概述：OneRetrieval 用单一生成式检索模型统一电商多路召回（稀疏/稠密/协同/生成），核心创新是 Keyword-Aligned Encoding（KAE）——把 identifier 位置与可解释关键词绑定，并在 codebook 中预留 reserved slots，让运营可在无需重训练的情况下将新词小时级绑定到目标商品集。
💡 创新性分析：首次将"可编辑性"作为生成式检索的结构性目标；KAE 让关键词与 identifier 显式对齐，使路由干预成为精确操作而非"碰运气"；线上 AB 证明可逐步替换各召回分支同时提升业务指标。
📊 关键指标：离线 Order HR@350=0.5482；线上 AB 替换倒排分支 Order +0.710%（显著）；替换几乎全部召回 Item CTR +0.821%（显著）。⭐ 总分：87/100

---

📄 标题：UNIVID: Unified Vision-Language Model for Video Moderation
👥 作者：Kejuan Yang, Yizhuo Zhang, Mingyuan Du, et al. (ByteDance)
🔗 链接：https://arxiv.org/abs/2606.05748
📝 方法概述：UNIVID 是字节跳动开发的统一视频审核 VLM，通过生成"策略感知字幕"作为可解释的中间表示，取代 1000+ 个政策特定分类器。三阶段 pipeline：(A) 风险漏斗过滤；(B) UNIVID-Lite（轻量决策）+ UNIVID-RAG（历史违规召回）；(C) Trend Governance（缓存嵌入追踪新兴风险）。训练数据结合专家精标与合成数据。
💡 创新性分析：以"字幕生成"取代"直接分类"是内容治理范式转变；单 backbone 替代千模型生态显著降低维护成本；策略感知字幕使审核决策人类可验证，支持跨任务复用；UNIVID-RAG 将历史违规写入检索减少漏检。
📊 关键指标：工业部署对比：违规漏检率相对降低 42.7%，误杀率相对降低 37.0%，模型数量从 1000+ 降至 1。⭐ 总分：86/100（含代码复现：code/UNIVID/）

---

📄 标题：Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution
👥 作者：Dimitri Kachler, Damien Sileo, Pascal Denis (Inria / Université de Lille)
🔗 链接：https://arxiv.org/abs/2606.13668
📝 方法概述：Influcoder 解决大规模数据归因/过滤的效率瓶颈：用 LoRA 子空间梯度 + CountSketch 压缩生成 ground-truth influence ranking，再训练 encoder 将样本映射为 influence embedding，之后用 embedding 余弦相似度替代梯度计算进行快速归因，实现 70× 加速、10× 存储降低，AUPRC 仅下降约 1.4%。
💡 创新性分析：以"influence 排序"而非 loss/输出为蒸馏目标，保留了因果语义；LoRA+CountSketch 双重压缩使 ground-truth 生成在单卡可行；推理期无梯度，embedding 可预计算持久化为数据归因服务。
📊 关键指标：毒性过滤 AUPRC=0.6981（vs LESS 0.7075）；推理时间 32.1s vs 2248.5s（70× faster）；存储 30MB vs 320MB（10× less）。⭐ 总分：82/100（含代码复现：Influcoder/）

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Alibaba International Digital Commerce Group
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：QueryAgent-R1 是记忆增强的电商查询推荐 Agent，通过 Chain-of-Retrieval Optimization 在 RL 训练过程中嵌入真实商品检索验证，使 Agent 能根据实际检索结果调整查询。一致性奖励函数联合优化 CTR（相关性）和 CVR（转化），Memory Abstraction Module 实现高效个性化用户画像。
💡 创新性分析：把"检索验证"内化为 RL 训练链路（training-time grounding）而非仅推理期调用；双目标一致性奖励弥合 CTR/CVR gap；首次从 Agent RL 角度解决电商查询推荐的"高点击低转化"问题。
📊 关键指标：线上 AB 实验显示 CTR 和 CVR 均显著提升（Alibaba 国际实际部署）。⭐ 总分：74/100

---

📄 标题：SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants
👥 作者：Yimin Hu, Mengtao Xu, Hao Guo, et al. (Alibaba Group)
🔗 链接：https://arxiv.org/abs/2606.12984
📝 方法概述：SkillChain 面向电商图片助手的多意图场景，将每个意图的行为约束显式化为声明式 Skill 规范，通过三阶段闭环自动演进：Stage-1 人工反思生成初始 Skill；Stage-2 路由失败挖掘治理 intent routing drift；Stage-3 双路径评估（规则+LLM-Judge）+ 聚合归因修复 Skill Body，实现单调质量提升。
💡 创新性分析：声明式 Skill 规范使工具调用和格式输出可控；Route Optimizer 专门治理工业常见但学术鲜有关注的路由漂移问题；双路径评估将线上失败自动转化为修复指令，闭环无需人工干预。
📊 关键指标：Full SkillChain 离线 LLM-Judge Avg=72.2；线上 AB Interactive UV +1.92pp，Full-read Rate +4.98pp，Avg Dwell Time +2.85s，7-day Return Rate +1.15pp。⭐ 总分：78/100

---

📄 标题：Iterating Toward Better Search: A Two-Agent Simulation Framework for Evaluating Agentic Search Architectures in E-Commerce
👥 作者：Jetlir Duraj, Jayanth Yetukuri, Shuang Zhou, et al. (eBay Inc.)
🔗 链接：https://arxiv.org/abs/2606.12924
📝 方法概述：eBay 提出可替换双 Agent 仿真框架评测对话购物助手：buyer agent（persona/mission/patience 固定）与 responder agent（接真实 eBay 搜索 API，可插拔）组合，使同一场景下不同架构可控对比。实验发现简单 rolling-window memory 优于复杂 intent-extraction memory，Judge 选择对结论有显著影响。
💡 创新性分析：Buyer 固定 / Responder 可插拔的解耦设计实现真正受控的架构对比；系统化 failure taxonomy 将低分样本转化为具体修复点；发现并量化 judge 分歧对评测结论的影响，为工业评测治理提供直接依据。
📊 关键指标：Sys-B vs Sys-A：Mission Success +0.10，SRP Relevance +0.07，Latency -35%；迭代后 near-failure 减少 62%，catastrophic failure 减少 36%。⭐ 总分：78/100

---

📄 标题：EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments
👥 作者：Jundong Xu, Qingchuan Li, Jiaying Wu, et al. (NUS / SMU / UW / UCL / UPenn / MIT)
🔗 链接：https://arxiv.org/abs/2606.13681
📝 方法概述：EvoArena 构建"版本演化链"评测 benchmark（Terminal/Software/Social 三域），发现现有 Agent 在持续演化环境中平均 accuracy 仅 39.6%。EvoMem 以 git-like patch history（before/after/rationale/evidence）记录记忆演化，默认用最新状态，冲突时回溯 patch 历史，改善版本感知推理。
💡 创新性分析：首次将"持续环境演化"建模为可量化 benchmark（release chain + chain-level accuracy）；EvoMem 的 patch history 结构使记忆更新具有可追溯性和可解释性；chain-level accuracy 揭示累积失败模式，对真实生产系统（工具/政策不断变更）有直接启示。
📊 关键指标：EvoMem 平均提升 +1.5%（EvoArena），+3.7%（chain-level），+6.1%（GAIA），+4.8%（LoCoMo）。⭐ 总分：68/100

---

📄 标题：Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT)
👥 作者：Zilin Xiao, Qi Ma, Chun-cheng Jason Chen, et al.
🔗 链接：https://arxiv.org/abs/2606.13680
📝 方法概述：RA-RFT 将 RAG 的检索目标从"语义相似"转为"推理效用"（reasoning utility），用 LLM judge 蒸馏训练 reasoning-aware retriever，在 RLVR 训练期注入高质量类比推理轨迹（training-time RAG，非仅推理期），提升探索效率与奖励密度。
💡 创新性分析：检索目标定义从语义转为推理效用，是 RAG+RL 融合的关键创新点；训练期而非推理期集成 RAG，使模型真正学会利用类比策略，而非每次推理时临时调用。
📊 关键指标：Qwen3-1.7B：RA-RFT vs GRPO，AIME24 55.1 vs 50.4（+4.7），AIME25 48.7 vs 41.6（+7.1）；Qwen3-4B：AIME25 69.2 vs 66.4（+2.8）。⭐ 总分：67/100

---

📄 标题：Multi-Agent RL from Delayed Marketplace Feedback for Three-Sided Dispatch (DoorDash)
👥 作者：Haochen Wu, Yi Hou, Ryan Xie (DoorDash)
🔗 链接：https://arxiv.org/abs/2606.13604
📝 方法概述：DoorDash 三方配送市场的 MARL 系统：RL 不替换组合优化器，而是让 store-level agent 选择离散 objective-weight multiplier（速度 vs 拼单），用延迟市场反馈（区域结果）做 offline RL，保守正则缓解 OOD，Switchback 实验严格验证。
💡 创新性分析：窄接口（weight multiplier）设计让高风险生产系统安全引入 offline RL；区域级延迟反馈奖励体现市场网络效应；DDQN + 保守正则 + switchback 三重保险确保部署安全。
📊 关键指标：CAT -1.261s (p=0.019)，CWT -0.856s (p=0.004)，% batched +0.495pp (p<0.001)，Dinner: CWT -1.030s (p=0.041)，%20-min late -0.037pp (p=0.040)。⭐ 总分：67/100
