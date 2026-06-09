# Feishu Cards — 2026-06-08

---

📄 标题：UNIVID: Unified Vision-Language Model for Video Moderation
👥 作者：Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang et al.（ByteDance）
🔗 链接：https://arxiv.org/abs/2606.05748
📝 方法概述：用单一政策感知 VLM 主干（LLaVA-OV 架构）替代千余个违规专项检测模型。模型输入视频+审核政策文本，输出结构化违规描述字幕（Policy-Aware Caption）及其嵌入向量，驱动三阶段审核流水线：①Risk Filter（高吞吐嵌入过滤）、②Moderation Actor（精/召两种精调变体）、③Trend Governance（字幕嵌入聚类发现新兴违规话题）。训练采用人工标注与高质量合成数据混合策略。
💡 创新性分析：首次将政策感知字幕作为跨阶段可复用的中间表示，将可解释性与效率统一；单一主干取代 1000+ 模型，工程维护成本大幅下降；趋势发现模块利用缓存字幕嵌入自动聚类，解决违规话题时效性问题。
📊 关键指标：TikTok 生产环境 — 漏判率（Violation Leakage）相对降低 42.7%，误判率（Overkill Rate）相对降低 37.0%；替换超过 1000 个策略专项模型。

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun 等（Alibaba International Digital Commercial Group）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：针对电商查询推荐中"高 CTR 低 CVR"问题，提出 QueryAgent-R1 记忆增强 Agentic RL 框架。Agent 先生成候选查询，随即调用真实商品检索引擎（Chain-of-Retrieval）观察结果并精炼查询；一致性奖励函数（Consistency Reward）同时优化查询点击率与商品转化率，打通端到端对齐；用户长期行为以兴趣图（Interest Graph）抽象存储以支持高效个性化。
💡 创新性分析：Chain-of-Retrieval 将商品检索闭合进查询生成推理链，是将 R1 风格强化学习引入电商查询推荐的首批尝试；一致性奖励解决了传统方法 CTR 与 CVR 目标分离的结构性缺陷。
📊 关键指标：在 Alibaba International 数千万日活平台上线 A/B 实验，CTR 与 CVR 联合提升（具体数值见论文）；两个离线数据集（私有+公开）均优于基线。

---

📄 标题：OneReason Technical Report
👥 作者：OneRec Team（Kuaishou，83 名贡献者）
🔗 链接：https://arxiv.org/abs/2606.06260
📝 方法概述：系统报告快手 OneRec 生成式推荐系统（已覆盖短视频、直播、广告、电商）中"推理能力激活"的探索与发现。通过 OneRec-Think（Itemic Alignment + Reasoning Scaffolding + 定制 RL 奖励）尝试在 item token 推荐序列中注入 Chain-of-Thought，并在快手线上进行大规模 A/B 实验验证。
💡 创新性分析：首次公开报告"thinking mode 在生成式推荐中无显著优于 non-thinking mode"的工业级负向结论；揭示 item-only CoT 无法携带有效语义推理信息的本质原因；对整个生成式推荐研究社区具有重要方向指引价值。
📊 关键指标：快手线上 A/B — App Stay Time +0.159%，Watch Time +0.169%，Follows +0.431%；Amazon Beauty 离线 Recall@5 从 0.0460 提升至 0.0563（OneRec-Think）。

---

📄 标题：Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis
👥 作者：Bin Wen, Tien-Ping Tan（Universiti Sains Malaysia）
🔗 链接：https://arxiv.org/abs/2606.05713
📝 方法概述：将多模态情感分析从生成式解码改为判别式隐藏态读取：在 Qwen2.5-Omni-7B 的 Thinker 模块末层隐藏态接线性回归头，单次前向传播直接输出连续情感分值，规避自回归解码的结构性误配。采用 QLoRA（4-bit 量化+低秩适配）仅 1.14% 参数可训，单卡 RTX 5090（32GB）即可训练完整 7B 流水线。
💡 创新性分析：方法简洁有效，将情感回归从 token 生成空间迁移回连续向量空间；工程上使 7B omni-modal 模型的微调门槛大幅降低；可迁移至商品评论情感分析场景。
📊 关键指标：CMU-MOSI/MOSEI 数据集上与 SOTA 竞争；峰值显存 10-21 GB，训练参数量 1.14%（QLoRA）。
