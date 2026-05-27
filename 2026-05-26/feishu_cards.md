# 飞书卡片 — 2026-05-26 日报 (分数 ≥ 40 的论文)
# Feishu Cards — 2026-05-26 Daily AI Paper Inspection

---

📄 标题：ClaimDiff-RL: Fine-Grained Caption Reinforcement Learning through Visual Claim Comparison
👥 作者：Tianle Li, Xuyang Shen, Yan Ma et al.（香港中文大学 CUHK + MiniMax）
🔗 链接：https://arxiv.org/abs/2605.20278
📝 方法概述：提出以「原子声明差异」为奖励最小单元的图像描述 RL 框架。多模态评判器（ClaimDiff Judge）对候选描述逐声明与参考对比，验证每条差异是否为幻觉或遗漏，赋予开放词表错误类型与严重级别，聚合为可分别调控幻觉惩罚和遗漏惩罚的双分量奖励信号，首次使两类错误独立可测可调。
💡 创新性分析：首次将 RL 奖励粒度细化至原子声明级别，打破整体标量奖励模式；多维度超越 Gemini-3-Pro-Preview 细粒度能力；直接适用于电商商品描述质量管控、达人短视频字幕审核、内容核实等场景。综合评分：81/100
📊 关键指标：在 160 图人工诊断基准实现幻觉-遗漏 Pareto 前沿优化；多项公开 Captioning/VQA 基准多维超越 Gemini-3-Pro-Preview；已实现完整 PyTorch 复现 → code/ClaimDiff-RL/

---

📄 标题：Your Embedding Model is SMARTer Than You Think
👥 作者：待公开
🔗 链接：https://arxiv.org/abs/2605.24938
📝 方法概述：揭示标准单向量嵌入模型（如 CLIP、E5 等）内部天然潜藏多向量能力，通过推理时的 SMART 解码策略激活这些潜在多向量，无需重新训练多向量模型。在图文跨模态检索任务中，细粒度匹配精度显著提升，兼顾单向量的存储效率与多向量的检索质量。
💡 创新性分析：无需重训练即可激活已有模型的多向量能力，大幅降低多向量检索的部署成本；直接适用于电商以图搜款、同款检索、内容相似性匹配。综合评分：74/100
📊 关键指标：多模态检索基准细粒度 Recall@K 显著提升于单向量基线；属性级查询检索精度超越纯单向量方法（详细数值待完整论文公开）

---

📄 标题：ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning
👥 作者：Zuhao Yang, Kaichen Zhang, Sudong Wang, Keming Wu, Zhongyu Yang, Bo Li, Xiaojuan Qi, Shijian Lu, Xingxuan Li, Lidong Bing（EvolvingLMMs-Lab）
🔗 链接：https://arxiv.org/abs/2605.20342
📝 方法概述：首个端到端 RL 训练的并行视频工具调用多 Agent 框架。主 Agent 单轮并行发出多个时间窗口裁剪请求，权重共享子 Agent 分别处理后聚合回答。识别并命名"工具先验悖论"训练障碍，提出 PARA-GRPO 算法（可解析性锚定格式奖励 + 帧预算随机化）解决训练收敛问题。
💡 创新性分析：首创并行视频工具调用 RL 架构，解决顺序调用的错误传播和上下文污染问题；PARA-GRPO 是通用 RL 训练稳定性改进，适用范围广。综合评分：67/100
📊 关键指标：长视频 QA 准确率显著优于顺序调用基线；工具调用轮次从 N 轮降至 1 轮（并行）；已开源代码：https://github.com/EvolvingLMMs-Lab/ParaVT

---

📄 标题：Toward Native Multimodal Modeling: A Roadmap
👥 作者：Siyu An, Junru Lu, Junnan Dong et al.（共 21 位，腾讯优图实验室、清华大学、香港大学等）
🔗 链接：https://arxiv.org/abs/2605.25343
📝 方法概述：原生多模态建模（NMM）综述与路线图。形式化定义架构原生性，区分 Late-fusion / Mid-fusion / Early-fusion / Born-native 四级演化阶段。Born-native 状态实现所有模态在统一 Transformer 空间内的对称多对多理解与生成。梳理从当前 SOTA 到 Born-native 的设计挑战与路径。
💡 创新性分析：首次形式化"原生性"分类框架，填补领域术语和分类理论空白；为电商内容理解的 MLLM 架构选型提供系统性指导。综合评分：55/100
📊 关键指标：覆盖 20+ 主流 NMM 模型分析（Show-o, Janus, BAGEL, LWM 等）；4 级原生性分类框架；21 位多机构作者合作

---

📄 标题：Claw-Anything: Benchmarking Always-On Personal Assistants
👥 作者：待公开
🔗 链接：https://arxiv.org/abs/2605.26086
📝 方法概述：提出 Claw-Anything 基准，沿三维度扩展 Agent 上下文：①长视野活动历史（模拟数月用户活动）、②相互依赖的后端服务（日历/邮件/购物/地图多服务链路）、③跨设备 GUI+CLI 混合交互。通过多轮事件注入生成含噪声的复杂世界状态，评测"永远在线"个人助理的真实能力。
💡 创新性分析：三维度上下文扩展揭示现有 Agent 基准的重大盲点；购物 Agent、达人运营助手等电商场景可直接受益于该基准的评测框架。综合评分：50/100
📊 关键指标：当前最优 Agent 在 Claw-Anything 上成功率远低于人类，揭示"永远在线"助理能力的巨大差距；基准涵盖跨月历史+多后端服务+跨设备交互三类任务
