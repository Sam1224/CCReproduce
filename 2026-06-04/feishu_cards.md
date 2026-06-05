# 飞书卡片汇总 — 2026-06-04 (评分 ≥40，共 7 篇)

> 本文件为飞书卡片等价 Markdown，可直接粘贴到飞书富文本/卡片机器人。
> 当前环境未配置飞书 MCP，仅输出本文件，不调用飞书 API。

---

📄 标题：OCL: Organizational Control Layer — Governance Infrastructure at the Execution Boundary of LLM Agent Systems
👥 作者：Tianyu Shi, Yang Mo, Yiou Liu, Zhuonan Hao, Yin Wang, Wenzhuo Hu, Nan Yu, Meng Zhou, Jiangbo Yu (McGill / Purdue / UNSW / UCLA / NYU / Stevens / Aimaikj Research)
🔗 链接：https://arxiv.org/abs/2606.04306
📝 方法概述：在 LLM agent 生成动作提案之后、环境实际执行之前插入"组织控制层"（OCL），通过策略强制（policy enforcement）+ 升级机制（escalation）对提案进行合规拦截，无需修改底层 LLM。支持符号规则、ML 分类器、人工审核多种策略形式。在电商对抗谈判场景（AgenticPay）评估，兼容 GPT-4o、Claude 等多个前沿后端。
💡 创新性分析：将"执行边界治理"作为独立抽象，区别于 input/output guardrail 和 tool-use 过滤；提案-执行解耦架构可插拔到任意 agent 框架，对电商商家 agent、客服 agent 的安全部署具有直接价值。
📊 关键指标：AgenticPay 对抗谈判环境 — 不合规执行率从 88% 降至近 0%；合法任务成功率从 12% 提升至 96%（多个前沿 LLM 后端均验证）。**得分 80，已复现 code/OCL/**

---

📄 标题：E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs
👥 作者：Xianjie Liu, Yiman Hu, Liang Wu, Ping Hu, Yixiong Zou, Jian Xu, Bo Zheng (Alibaba / Taobao)
🔗 链接：https://arxiv.org/abs/2602.08355
📝 方法概述：提出首个专为电商短视频设计的 MLLM 评测基准 E-VAds，引入多模态信息密度评估框架量化视觉/音频/文本三模态复杂度，证明电商视频远高于通用视频。从淘宝抓取 3,961 个高质量视频，利用多智能体系统生成 19,785 个开放式问题，覆盖商业意图、商品属性、CTA 识别等维度。对 GPT-4o、Gemini、Claude、LLaVA 等主流 MLLM 进行系统评测。
💡 创新性分析：填补电商视频理解基准空白；多模态信息密度框架为 benchmark 设计提供量化工具；multi-agent 问题生成兼顾规模与质量，方法论可借鉴。
📊 关键指标：E-VAds 3,961 视频 / 19,785 Q&A；所有被测 MLLM 在高密度电商视频场景下出现明显性能退化；商业意图理解 GPT-4o 最优（具体数字见原文）。**得分 78**（*提交日期 2026-02，catch-up 发现*）

---

📄 标题：LARM: LLM-Alignment Live-Streaming Recommendation
👥 作者：Yueyang Liu, Jiangxia Cao, Shen Wang, Shuang Wen, Xiang Chen, Xiangyu Wu, Shuang Yang, Zhaojie Liu, Kun Gai, Guorui Zhou (Kuaishou Technology 快手)
🔗 链接：https://arxiv.org/abs/2504.05217
📝 方法概述：针对直播间不同时刻语义动态差异大的特点，提出 LARM 框架：① 在快手直播数据上微调开源 LLM，生成随时间变化的上下文感知 embedding；② 通过可学习 gate 机制将 LLM embedding 与 ID embedding 对齐融合；③ 转换为离散 semantic code，无缝接入现有召回与排序系统，无需重构基础设施。
💡 创新性分析：系统解决"直播 item 静态化"痛点；LLM 时序语义感知 + ID 对齐 + semantic code 三项创新相互协同；快手在线系统验证有说服力。
📊 关键指标：快手直播推荐系统在线显著提升（具体数字见原文）；三项创新通过逐项 ablation 验证各自有效性。**得分 72**（*提交日期 2026-04，catch-up 发现*）

---

📄 标题：TAP-PER: Beyond Retrieval — Learning Compact User Representations for Scalable LLM Personalization
👥 作者：Heng Cao, Fan Zhang, Jian Yao, Yujie Zheng, Changlin Zhao, Lu Hao, Yuxuan Wei, Wangze Ni, Huaiyu Fu, Yuqian Sun, Xuyan Mo
🔗 链接：https://arxiv.org/abs/2606.04547
📝 方法概述：提出 TAP-PER（Temporal Attentive Prefix for PERsonalization），以可学习 prefix 嵌入替代显式 prompt 构建和重量级 per-user adapter：长期偏好用 user-state prefix 编码，即时意图用 query-conditioned record prefix 动态激活，两者注入 LLM 的 attention key-value cache，结合时序信号捕捉兴趣演化。
💡 创新性分析：双前缀分解实现"长期偏好 + 即时意图"正交建模；per-user 仅存轻量 prefix 向量，支持亿级用户规模；无需修改 LLM 参数，与任意 frozen LLM 兼容。
📊 关键指标：在个性化基准上质量持平或优于检索式和 LoRA 基准，存储开销显著下降（具体数字见原文）。**得分 70**

---

📄 标题：RUBAS: Rubric-Based Reinforcement Learning for Agent Safety
👥 作者：（见原文 arXiv:2606.04051）
🔗 链接：https://arxiv.org/abs/2606.04051
📝 方法概述：将 agent 安全行为分解为四维度（工具调用安全、参数安全、响应安全、有用性），对应设计结构化 Rubric 作为细粒度奖励信号，在完整 agent trajectory 上执行 RL 优化，实现安全工具调用与任务完成的联合优化，无需修改 LLM 架构。
💡 创新性分析：四维度分解比粗粒度 refusal 信号更精确且可解释；Rubric 的规则可维护性高；适合电商商家 agent 等需要精细安全控制的场景。
📊 关键指标：多个 agent safety benchmarks 上优于标准对齐 baseline；工具调用 hallucination 显著下降；有用性维持竞争力（具体数字见原文）。**得分 62**

---

📄 标题：PHF: Beyond Isolated Behaviors — Hierarchical User Modeling for LLM Personalization
👥 作者：Liang Wang, Xinyi Mou, Xiaoyou Liu, Tiannan Wang, Yuqing Wang, Zhongyu Wei
🔗 链接：https://arxiv.org/abs/2606.02300
📝 方法概述：受社会学家布尔迪厄实践理论启发，提出 PHF 三层次用户建模框架：Practice（单次行为序列）→ Habitus（时序积累稳定偏好）→ Field（跨用户群体共性），三层次综合指导 LLM 的个性化输出，超越扁平化行为聚合范式。
💡 创新性分析：社会学理论迁移提供充分理论基础；Field 层引入群体级相似用户信息，类似 CF 但在 LLM 框架中实现；三层次 ablation 各有贡献。
📊 关键指标：多任务个性化基准上三层次各组件 ablation 正向（具体数字见原文）。**得分 62**

---

📄 标题：Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs
👥 作者：Wei Zhou, Jun Zhou, Haoyu Wang 等 19 人 (Tsinghua / Alibaba / Microsoft / ByteDance)
🔗 链接：https://arxiv.org/abs/2601.17058 ；GitHub: https://github.com/weAIDB/awesome-data-llm
📝 方法概述：系统梳理 LLM 用于数据准备的最新进展，覆盖数据清洗（标准化、错误修复、缺失填充）、数据集成（实体匹配、Schema 匹配）、数据增强（标注、画像、合成）三大任务，归纳从规则 pipeline 到 prompt 驱动 agent 化工作流的范式转移。
💡 创新性分析：首个系统性"LLM × 数据准备" survey，分类框架完整；重点关注 agentic 工作流的演化趋势；对电商商品数据清洗、属性标注 pipeline 设计具指导价值。
📊 关键指标：覆盖实体匹配 / Schema 匹配 / 缺失填充 / 标注等子任务主要 SOTA 对比（各子任务数字见原文）。**得分 58**（*提交日期 2026-01，catch-up 发现*）
