# Feishu Cards — 2026-06-22 Daily AI Paper Inspection

> 以下为评分 ≥ 40 的论文飞书卡片（共 8 篇），按评分降序排列。

---

📄 标题：Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation
👥 作者：Yuecheng Li, Zeyu Song, Jing Yao, Chi Lu, Peng Jiang, Kun Gai（快手科技）
🔗 链接：https://arxiv.org/abs/2606.03866
📝 方法概述：Taiji 是快手面向工业推荐的 LLM-as-Enhancer 框架，解决两大核心难题：(1) SFT瓶颈——通过逆向推理 CoT 数据生成让 LLM 学语义推理而非记忆 ID；(2) RL 对齐问题——提出 POPO（帕累托最优策略优化），动态调整语义奖励与协同 ID 奖励权重，理论保证帕累托最优权衡。
💡 创新性分析：POPO 是首次将帕累托理论引入 LLM 推荐 RL 对齐的工作，从理论层面解决语义-ID 多目标冲突；逆向 CoT 数据生成在避免 ID 记忆的同时保留了 LLM 的世界知识。
📊 关键指标：快手广告平台线上 A/B（4亿日活）：广告主价值（ADVV）+2.83%，整体营收 +3.30%。综合评分：84/100 ✦代码复现见 code/Taiji/

---

📄 标题：Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations
👥 作者：Zhuohang Jiang, Yuxin Chen, Shijie Wang, Haohao Qu et al.（内容-电商平台，KDD 2026）
🔗 链接：https://arxiv.org/abs/2606.10357
📝 方法概述：AIR 针对内容-电商跨域推荐难题（语义鸿沟大、行为序列噪声高、LLM 在线延迟不可接受），提出"原子意图"抽象：先用轻量模型从用户行为序列中蒸馏高信号购买意图描述，再通过 LLM 离线推理跨域语义，最后通过知识蒸馏注入在线推荐模型，实现"LLM思考+轻模型执行"架构。
💡 创新性分析：首次引入原子意图压缩层桥接内容-电商语义鸿沟；离线 LLM + 在线蒸馏的架构设计有效规避延迟瓶颈，已在 KDD 2026 顶会通过审稿。
📊 关键指标：CTR/CVR 双指标显著提升（KDD工业Track标准验证），线上 A/B 正向；具体数值受 NDA 保护。综合评分：83/100 ✦代码复现见 code/AIR/

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng（阿里巴巴国际数字商业集团）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：QueryAgent-R1 解决电商查询推荐"高 CTR、低 CVR"的核心矛盾。核心创新是链式检索优化（Chain-of-Retrieval）：Agent 先生成候选查询词，再调用真实商品检索引擎验证商品匹配性，基于检索结果精炼查询词；RL 训练时用一致性奖励联合优化 CTR+CVR；记忆抽象模块维护用户偏好画像。
💡 创新性分析：将 R1 式 RL 思路（思考-验证-精炼）迁移至电商查询推荐；引入商品库存检索作为 ground truth 约束，从根本上解决生成词与库存脱节问题。
📊 关键指标：阿里国际电商平台离线+线上 A/B 验证，CTR+CVR 综合提升（具体数值未公开）。综合评分：76/100

---

📄 标题：SSRLive: Live Streaming Recommendation with Dynamic Semantic ID
👥 作者：Teng Shi, Zhaoheng Li, Yuanhang Qu, Yi Liu, Lixiang Lai, Yuning Jiang（阿里巴巴淘宝天猫集团）
🔗 链接：https://arxiv.org/abs/2606.06970
📝 方法概述：SSRLive 为直播推荐设计静态+动态语义 ID 统一框架。生成模块（编码器-解码器）生成静态 SID（稳定风格）和动态 SID（多模态实时内容：视频+音频+弹幕）；判别模块融合 SID 与用户互动数据（关注/礼物/评论）做多任务预测，提升计算资源利用率（FLOPs）。
💡 创新性分析：首次将生成式语义 ID 扩展至动态变化的直播场景，实现"内容随时间变化"的实时表征更新；生成-判别双模块联合训练优化端到端效果。
📊 关键指标：淘宝天猫直播线上 A/B：观看时长 +3.38%，GMV +0.72%，粉丝增长 +3.12%，互动量 +2.92%。综合评分：75/100

---

📄 标题：CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning
👥 作者：Penghui Yang, Long Xing, Xiaoyi Dong et al.（InternLM / Shanghai AI Lab / CUHK）
🔗 链接：https://arxiv.org/abs/2606.09393
📝 方法概述：CapRL++ 将可验证奖励强化学习（RLVR）引入图像/视频 Dense Captioning：LVLM 生成描述 → 描述送给 vision-free LLM 回答 MCQ 问题 → MCQ 准确率作为可验证奖励信号（无需人工标注）。图像/视频统一框架，视频增加时序格式奖励和长度正则化。
💡 创新性分析：首次将 RLVR 从封闭推理任务扩展至开放式 Captioning；可验证奖励"描述能帮盲模型答对视觉题"从信息论角度定义了描述质量；代码已开源。
📊 关键指标：在 20+ 图像和视频 benchmark 全面提升；Prism 框架评估紧凑模型可媲美 Qwen2.5-VL-72B、Qwen3-VL-235B 超大模型。综合评分：75/100

---

📄 标题：G2Rec: Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation
👥 作者：Ruizhong Qiu, Yinglong Xia, Dongqi Fu et al.（UIUC + Meta Monetization Recommendation Systems）
🔗 链接：https://arxiv.org/abs/2606.20554
📝 方法概述：G2Rec 解决生成式推荐中物品 Tokenization 无法同时整合图结构协同信号和物品语义的问题。核心是图感知 co-engagement 建模（用 GNN 将"共同被参与的物品"关系注入 token embedding）+ 分布式用户兴趣结构化（显式建模多峰兴趣分布），在 Meta MRS 工业系统验证。
💡 创新性分析：将图神经网络的协同信号系统性整合进生成式推荐 Tokenization；用户兴趣多峰分布建模有效缓解"单一向量"无法表达复杂偏好的问题。
📊 关键指标：Meta MRS 工业数据集 Recall@K 和公开 benchmark NDCG 显著优于 TIGER/SASRec baseline。综合评分：65/100

---

📄 标题：From Content to Knowledge: Lightning Fast Long-Video Understanding with Neural Knowledge Representations
👥 作者：Yuchen Guan, Xiao Li, Zongyu Guo et al.（Microsoft Research Asia / Tsinghua）
🔗 链接：https://arxiv.org/abs/2606.11913
📝 方法概述：NKR 提出将视频"内化"为网络权重而非 token 序列：每段视频对应一组小型网络权重（附加到 VLM backbone），通过 Agentic Knowledge Distillation（AKD）—— Agent 自动合成视频的密集描述和问答对 —— 将视频知识蒸馏进权重。一次蒸馏后，后续对同一视频的任意查询可"闪电"推理。
💡 创新性分析：参数化视频内化范式在长视频推理中将计算量从 O(视频长度) 降至 O(参数规模)，多次查询同一视频时边际成本趋零，对内容审核多轮分析场景有重要价值。
📊 关键指标：EgoSchema 等视频 QA benchmark 上优于帧采样 baseline，推理速度显著提升。综合评分：60/100

---

📄 标题：Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection
👥 作者：Aswathy Velutharambath, Neele Falk, Sofie Labat, Tarun Tater, Amelie Wührl（欧洲 NLP 研究机构）
🔗 链接：https://arxiv.org/abs/2606.04924
📝 方法概述：对 155 名 NLP 研究者的社区调查，量化 LLM 对众包数据质量的冲击：调查众包工作者使用 LLM 代替人工完成标注任务的现状、检测策略（文本风格模式、异常完成速度）和对数据质量的系统性影响。
💡 创新性分析：44% 受访者已发现数据集中有 LLM 痕迹；揭示当前检测手段不足，为内容平台的人工数据采集质量保障提供警示和方向参考。
📊 关键指标：N=155 社区调查；44% 已观察到 LLM 使用；93% 预料到但 50% 不知如何应对。综合评分：44/100
