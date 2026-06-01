# 飞书卡片 — 2026-05-31 每日论文速报

---

📄 标题：Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching
👥 作者：ByteDance/直播内容安全团队（KDD 2026）
🔗 链接：https://arxiv.org/abs/2512.03553
📝 方法概述：提出双路径混合架构，监督分类路径处理已知违规，基于 MLLM 知识蒸馏的相似度检索路径处理新兴/模糊违规，两路共享 text/audio/visual 三模态输入，联合决策。
💡 创新性分析：将 MLLM 作为教师蒸馏到轻量 re-ranker，并引入检索路径解决分类器对新违规类型泛化差的根本问题；三模态对齐使语音话术、视觉展示同时纳入审核。
📊 关键指标：相似度路径 76% recall @ 80% precision；混合框架显著超越单路 baseline（生产部署数据）

---

📄 标题：Valley3: Scaling Omni Foundation Models for E-commerce
👥 作者：Zeyu Chen, Guanghao Zhou, Qixiang Yin 等（ByteDance Group）
🔗 链接：https://arxiv.org/abs/2605.01278
📝 方法概述：在 Qwen3-VL 骨干基础上接入 Audio Transformer，通过四阶段电商连续预训练（音频理解→跨模态指令跟随→电商领域知识→长文本推理）构建全感官电商大模型，配备可控推理（非思考/三级思考）与 Agentic Search 能力。
💡 创新性分析：首个原生支持多语言音频的电商 omni MLLM；可控推理平衡效率与深度；Agentic Search 支持主动检索电商规则库，开启 AI 原生电商助理范式。
📊 关键指标：omni 电商 benchmark 6 个任务全面超越对比 baseline，通用 benchmark 保持竞争力；代码/权重已开源

---

📄 标题：Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment
👥 作者：Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He（中科院计算所）
🔗 链接：https://arxiv.org/abs/2601.16027
📝 方法概述：提出 CS-VAR（跨场次证据感知 RAG 检测器），用向量检索从历史 session 库中取回相似违规场次，结合 LLM 进行结构化跨场次推理，再通过知识蒸馏将推理能力迁移至轻量小模型，实现实时部署。
💡 创新性分析：首次将跨场次 RAG 引入直播风险检测，识别"剧本重复"的诈骗团伙；LLM 教师→小模型蒸馏解决延迟与推理质量的两难。
📊 关键指标：真实平台数据上显著超越单 session baseline（AUC 和 Precision@K），小模型支持实时推理

---

📄 标题：TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation
👥 作者：（工业界电商研究团队）
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：提出 TGQ-Former，以商品结构化元数据为语义引导，通过双路 Hybrid-Query Connector（元数据引导路 + 探索路）提取干净视觉 token，并用可靠性感知的双门控向量调制（DGVM）自适应融合两路输出，消除促销贴纸/背景噪声对 I2I 检索的干扰。
💡 创新性分析：针对电商图片高噪声场景，将文本语义引导嵌入视觉 token 提取过程，无需微调视觉编码器；DGVM 动态降噪机制独特。
📊 关键指标：大规模真实电商全池 I2I 检索 Hit Rate@100 平均提升 +6.04%

---

📄 标题：RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation
👥 作者：（内容安全研究团队）
🔗 链接：https://arxiv.org/abs/2605.07760
📝 方法概述：构建包含 93 条原子规则和 92 个类型化规则关系的内容审核诊断框架，将审核决策拆解为规则激活识别→规则交互恢复→决策充分性判断→结论解析四步链，系统评估现有 VLM 在规则推理中的弱点。
💡 创新性分析：从"判定结果准确率"转向"推理链诊断"，暴露 VLM 在跨规则交互推理上的核心短板，为内审模型定向优化提供依据。
📊 关键指标：93 规则 + 92 关系 benchmark；多数 VLM 在跨规则交互推理任务上性能显著下降

---

📄 标题：A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems
👥 作者：Yiming Zhu, Xu Liu, Ziyun Xu, Zheng Wu 等（Meta Platforms）
🔗 链接：https://arxiv.org/abs/2605.09338
📝 方法概述：提出离线 MLLM 多媒体语义提取 + 在线轻量特征消费的两阶段框架，解决 MLLM 在延迟敏感、亿级 item 规模推荐系统中的部署挑战，在 Meta 生产推荐系统中验证有效性。
💡 创新性分析：将 MLLM 计算成本转移至离线阶段，在线服务仅消费紧凑的语义特征——这是当前大规模推荐系统中 MLLM 落地的可行工程路径。
📊 关键指标：Meta 内部大规模推荐系统 A/B 测试中显著提升用户参与度指标，满足工业延迟 SLA

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis（Fastino AI）
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：提出 0.3B 双向编码器（基于 GLiNER2）的 LLM 安全防护方案，将任务定义和标签语义以 schema token 编码入输入序列，单次前向传播同时完成 14 类细粒度伤害分类 + 多安全维度判定，无需自回归解码。
💡 创新性分析：将 guardrail 从自回归文本生成还原为判别分类，带来 16x 速度提升、17x 延迟降低、23-90x 模型压缩；schema token 设计支持零代码扩展新安全维度。
📊 关键指标：准确率持平 7B-27B 对比模型，推理速度 16x，延迟 17x 更低，模型仅 0.3B
