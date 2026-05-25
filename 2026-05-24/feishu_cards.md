# Feishu Cards — 2026-05-24 Daily Report
> Papers with score ≥ 40 (7 papers)

---

📄 标题：RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation
👥 作者：Zhifeng Lu, Dianyuan Wang, Yuhu Shang, Zhenbo Xu
🔗 链接：https://arxiv.org/abs/2605.07760
📝 方法概述：构建首个基于规则条件推理的视觉语言内容审核基准。从公开平台内容政策中提炼 93 条原子规则和 92 种规则关系，生成 2,166 个图文案例，覆盖仇恨言论、成人内容、欺诈内容三大高风险政策族，评估 VLM 是否真正理解政策规则而非依赖表面特征进行判断。
💡 创新性分析：现有多模态安全基准将内容审核简化为标签匹配，忽略了平台政策的规则结构。RuleSafe-VL 首次将规则关系（条件/排斥/优先）形式化为有向图，测试规则激活、规则交互、证据充分性三层推理能力，揭示当前顶级 VLM 在政策驱动审核中的系统性缺陷。
📊 关键指标：主流 VLM（GPT-4V、Gemini 1.5 Pro 等）规则条件决策端到端 F1 低于 60%，人类基线 >85%；规则排斥/优先交互推理失败率显著偏高。评分：80/100。

---

📄 标题：Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation (TGQ-Former)
👥 作者：（见论文正文）
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：提出文本引导的 Q-Former（TGQ-Former），用结构化商品元数据（标题、类目、属性）作为语义锚，通过 cross-attention 指导视觉 token 提取，同时保留探索性视觉流捕捉细粒度外观特征；引入可靠性感知双门调制模块，根据图像噪声程度自适应融合两路视觉信号。
💡 创新性分析：标准 Q-Former 在含噪声促销贴图的电商图像上效果退化严重。TGQ-Former 首次将商品结构化文本作为 Q-Former 的语义约束，双路设计兼顾语义对齐与视觉发现，双门调制在推理时动态适应图像质量，工程实用性强，可部署于亿级商品全库检索系统。
📊 关键指标：大规模真实电商数据集全库检索，Hit Rate@100 (H@100) 平均提升 6.04%，优于强连接器基线和端到端 MLLM 基线。评分：79/100。

---

📄 标题：A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems
👥 作者：Yiming Zhu, Xu Liu, Ziyun Xu, Zheng Wu, Joena Zhang, Sirius Chen 等（Meta Platforms）
🔗 链接：https://arxiv.org/abs/2605.09338
📝 方法概述：提出三层工业级框架，将多模态大语言模型（LLaMA2-based）的语义理解能力注入实时推荐系统：(1) 离线 MM-LLM 层对内容生成丰富 caption 和语义嵌入；(2) 语义蒸馏层将 LLM 能力压缩到轻量在线模型；(3) 在线推荐层满足延迟约束。在 Meta 大规模系统上验证。
💡 创新性分析：LLM 直接用于在线推荐延迟不可接受，本框架将 LLM 定位为"离线语义标注器"，通过 caption 蒸馏将语义能力注入实时系统，提供了工业 MM-LLM 集成的实用范式，来自 Meta 的真实系统验证具有高可信度。
📊 关键指标：Meta 大规模推荐系统用户参与率显著提升；在线推理延迟 <10ms；优于纯轻量视觉编码器基线。评分：72/100。

---

📄 标题：Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework
👥 作者：Ewelina Gajewska 等
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：提出三层多智能体个性化内容审核框架：专家智能体层（多个领域 LLM 专家各司其职）+ 管理智能体层（协调路由与综合判断）+ 幽灵档案智能体层（基于用户人口统计学和文化背景模拟用户视角）。三层协同将统一政策转化为个性化内容判断。
💡 创新性分析：一刀切内容审核导致误杀合规内容和漏审违规内容并存。"幽灵档案智能体"是核心创新：通过模拟而非直接使用用户数据实现个性化，保护隐私；专家-管理-档案三层分离使系统模块化，各层可独立升级。在内容审核领域首次系统引入 RLHF 式个人偏好对齐思路。
📊 关键指标：个性化内容审核准确率相比统一政策基线提升 **32%**；用户敏感性对齐度显著改善。评分：68/100。

---

📄 标题：DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection
👥 作者：Junchao Wu, Yefeng Liu, Chenyu Zhu, Hao Zhang, Zeyu Wu, Tianqi Shi, Yichao Du, Longyue Wang, Weihua Luo, Jinsong Su, Derek F. Wong
🔗 链接：https://arxiv.org/abs/2605.15518
📝 方法概述：构建迄今最全面的多语言 LLM 文本检测基准 DetectRL-X：8 种语言 × 6 个高风险领域（含电商评论）× 4 种商业 LLM × AI 辅助写作操作（润色/扩写/压缩），共 8 个评估维度，系统揭示当前检测器在真实多语言商业场景下的可靠性瓶颈。
💡 创新性分析：现有 AIGC 文本检测基准以英语为主，忽视人机协作写作场景（AI 辅助润色而非完全机器生成）。DetectRL-X 首次全面覆盖商业 AI 辅助写作，直接对应虚假商品评论、AI 生成商品描述、达人营销文案等电商违规场景。
📊 关键指标：跨语言场景下现有检测器 AUC 普遍 <0.70；AI 辅助写作（尤其是润色）检测显著难于完全机器生成文本；跨语言预训练模型整体表现最优。评分：61/100。

---

📄 标题：A Bitter Lesson for Data Filtering
👥 作者：Christopher Mohri, John Duchi, Tatsunori Hashimoto（Stanford University）
🔗 链接：https://arxiv.org/abs/2605.19407
📝 方法概述：通过系统性 scaling 实验研究数据过滤对大模型预训练的影响，重点考察高计算量+数据稀缺的训练场景。对比无过滤、不同强度过滤、极端低质数据（token 打乱、随机注入）等多种条件，分析过滤强度与计算量的交互效应。
💡 创新性分析：挑战"高质量数据过滤是 LLM 必要条件"的行业共识。核心发现：在计算资源充裕但数据有限时，充分训练的大参数量模型不仅能容忍低质量数据，甚至能从"坏数据"中获益。揭示了"数据过滤有益"这一信念的适用边界，对电商域 LLM 预训练策略有直接指导价值。
📊 关键指标：高计算场景下"无过滤"≥"有过滤"（困惑度和下游任务均成立）；注入大量随机 token 序列不比同等数据量重复 epoch 显著更差；低计算场景边界条件仍支持过滤。评分：60/100。

---

📄 标题：Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems
👥 作者：43 位共同作者（多机构）
🔗 链接：https://arxiv.org/abs/2605.18747
📝 方法概述：综述论文，从"代码即线束（Harness）"视角统一梳理代码在 LLM 智能体中的三大角色：可执行推理（CoT→代码）、可编程行动（工具调用→程序接口）、可检验环境状态（中间结果验证）。构建线束接口、机制、可靠性三层调研框架，覆盖规划、记忆、工具使用、反馈控制等核心机制。
💡 创新性分析：首次以统一"线束"框架系统梳理代码在 LLM 智能体生态中从"输出目标"到"执行基础"的范式转变；43 位作者的大范围调研覆盖最新进展；识别长序列稳定性、验证完整性等开放挑战。对电商内容生成、审核流水线自动化的智能体架构设计有参考价值。
📊 关键指标：综述论文，无直接实验指标；GitHub 代码库收录相关论文列表（https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers）。评分：42/100。
