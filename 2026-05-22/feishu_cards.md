# Feishu Cards — 2026-05-22 Daily Inspection

> 共 6 篇论文，全部评分 ≥ 40，均生成飞书卡片。

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis（Fastino Labs）
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：GLiGuard 是一个 0.3B 参数的 schema-conditioned 双向编码器，将 LLM 内容安全审核统一为结构化分类任务。将任务名称和候选标签作为 token schema 嵌入输入序列，通过单次前向传播同时评估提示安全性、响应安全性、越狱检测和细粒度危害类别，无需多模型或多 prompt 模板切换。
💡 创新性分析：改变了 guardrail 的范式——从自回归解码器（7B-27B）转向双向编码器（0.3B），在 schema 条件分类框架下统一多任务审核。schema-conditioned 推理方式使模型具备零样本扩展到新安全类别的能力。
📊 关键指标：9 个安全基准 F1 与 7B-27B 模型相当；推理吞吐 **16× 提升**；延迟 **17× 降低**；模型比竞争对手小 **23-90×**。评分：**82/100**。

---

📄 标题：Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation
👥 作者：（作者信息待索引）
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：TGQ-Former（Text-Guided Q-Former）通过混合查询连接器将商品图片视觉 token 流分为"元数据锚定流"和"自由探索流"两路，解决电商图片中促销遮挡和背景噪声导致的检索精度下降问题。轻量级可靠性感知双门控向量调制模块自适应校准两路输出的贡献权重。
💡 创新性分析：首次将结构化商品元数据（品名、类目、属性）显式引入 Q-Former 查询生成，与自由探索查询解耦，是 MLLM 推荐连接器设计的真实架构创新，直接针对工业电商图片噪声问题。
📊 关键指标：大规模真实电商数据全库检索 Hit Rate@100 平均提升 **6.04%**；一致性优于强连接器基线和端到端 MLLM。评分：**79/100**。

---

📄 标题：A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems
👥 作者：Yiming Zhu, Xu Liu, Ziyun Xu, Zheng Wu, Joena Zhang, Sirius Chen 等（Meta Platforms）
🔗 链接：https://arxiv.org/abs/2605.09338
📝 方法概述：提出三段式 MM-LLM 框架用于工业级推荐系统中的多媒体内容理解：(1) LLaMA2 + Q-Former 生成视频/图片描述字幕；(2) 字幕 token 化为 DLRM 兼容的类别特征；(3) 1.5B 精简 LLM 满足实时推理约束。框架已在 Meta 平台部署，为 UGC 内容生成实时字幕并注入推荐模型。
💡 创新性分析：明确解决了 MM-LLM 与工业 DLRM 推荐栈之间的工程接口问题；三段式解耦设计使各层可独立演进；发表于 SIGIR '26 工业轨道，Meta 规模验证。
📊 关键指标：Meta 平台实际部署验证，显著提升用户偏好建模保真度；SIGIR '26 接收（顶级 IR 会议工业轨道）。评分：**73/100**。

---

📄 标题：Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework
👥 作者：（作者信息待索引）
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：提出 LLM 多智能体个性化推理框架用于内容审核决策：领域专家智能体负责特定有害内容类别分析，管理智能体协调分析和专家选择，幽灵档案智能体模拟个体用户敏感度视角（含年龄、文化背景等），输出差异化的个性化审核决策。
💡 创新性分析：将内容审核的"治理正当性（legitimacy）"问题引入 LLM agent 框架，通过个体用户敏感度建模超越一刀切的平台政策，是内容治理研究方向的新视角，有助于电商达人内容管理的精细化运营。
📊 关键指标：框架验证以定性为主；聚焦治理正当性而非单纯准确率。评分：**60/100**。

---

📄 标题：Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents
👥 作者：Akshay Manglik 等（Scale AI）
🔗 链接：https://arxiv.org/abs/2605.21347
📝 方法概述：Insights Generator（IG）是一个多智能体系统，对 LLM agent 执行轨迹语料库进行系统性诊断。通过假设提出-验证循环，跨 trace 群体发现规律性行为模式，生成有证据支撑的自然语言洞察报告，解决人工抽样检查无法发现种群级别规律性失败的问题。
💡 创新性分析：将 agent 调试提升至语料库级别（corpus-level）是有价值的工程化贡献；假设驱动分析确保洞察的可信度。可应用于电商搜索/推荐/审核 agent 的系统性 debug。
📊 关键指标：定性（报告质量评分）+ 定量（实施洞察后的下游性能提升）双维度评估。评分：**58/100**。

---

📄 标题：SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution
👥 作者：Hongyi Liu, Haoyan Yang, Tao Jiang, Bo Tang, Feiyu Xiong, Zhiyu Li
🔗 链接：https://arxiv.org/abs/2605.18401
📝 方法概述：SkillsVote 是 LLM agent 技能的全生命周期治理框架，将 skills 定义为耦合可执行脚本与程序性指引的经验模式。对百万级开源语料库进行环境需求、质量和可验证性画像，合成验证任务，并通过执行前 agentic 库搜索为 agent 提供高质量结构化技能上下文，防止低质量 skill 污染知识库。
💡 创新性分析：将 agent skill 的管理从单次收集延展到"收集→推荐→演进"全生命周期，具有治理（governance）视角；可验证性过滤对建立可信 agent 基础设施至关重要。
📊 关键指标：百万级开源语料库画像；可验证性技能过滤有效降低低质量 skill 污染率。评分：**55/100**。
