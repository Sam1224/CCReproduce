# Feishu Cards — 2026-06-19

> 域：电商内容生态 · 达人治理 | 共 4 篇（分数 ≥ 40）

---

📄 标题：PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models
👥 作者：Yueyi Sun, Yuhao Wang, Jason Li et al. (Peking University MSALab · ByteDance)
🔗 链接：https://arxiv.org/abs/2606.19534 | 代码：https://github.com/MSALab-PKU/PerceptionDLM
📝 方法概述：PerceptionDLM 将扩散语言模型（LLaDA-8B）天生的 token 级并行性用于**并行区域描述**。在 SigLIP-2 视觉编码器基础上，通过三个核心组件——区域提示（每掩码区域一个可学习嵌入绑定身份）、RoI 对齐特征回放（将池化区域特征重注入去噪流）、结构化块状注意力掩码（强制区域 token 只关注自身信息）——实现一次扩散去噪同时输出所有掩码区域描述，彻底消除自回归串行瓶颈。配套提出首个同时评测质量与效率的 ParaDLC-Bench 基准。
💡 创新性分析：首次将扩散 LM 的并行性系统用于多区域细粒度 captioning；三组件形成闭环缺一不可；ParaDLC-Bench 填补评测空白；字节达到工业落地规模，区域级描述直接映射电商商品图像多属性标注场景。综合评分 **88/100（STRONG）**，官方代码已开源。
📊 关键指标：ParaDLC-Bench 准确率 62.4%（LLaDA-V 35.2%，SDAR-VL 31.3%）；吞吐 TPF=2.9（基线 1.0）；单图 4 masks 延迟 2.92 s vs 10.04 s（3.44×↑）；16 个 MLLM 基准 15/16 超越 LLaDA-V。

---

📄 标题：Non-negative Elastic Net Decoding for Information Retrieval
👥 作者：Koki Okajima, Yasutoshi Ida, Tsukasa Yoshida, Yasuaki Nakamura (NTT, Inc.)
🔗 链接：https://arxiv.org/abs/2606.17910
📝 方法概述：NNN（Non-negative elastic Net decoding）将稠密检索从"逐文档独立内积打分"重构为**联合稀疏非负重建**问题：给定查询嵌入 q 和文档矩阵 D，用弹性网优化 min_w ½‖q-Dw‖²+α‖w‖₁+β‖w‖₂² s.t. w≥0，支撑集即为检索结果。文档之间在权重层面相互竞争与互补，减少冗余近重复文档、提升集合完整性。FISTA 求解器可直接作用于冻结嵌入（仅 2 个超参），或展开为端到端训练。论文同时给出 NNN 严格包含稠密检索的理论分离定理。
💡 创新性分析：问题重构视角新颖，将检索从点积打分变为嵌入重建；严格分离定理在检索领域罕见；零参数适配冻结嵌入即可使用；+36% completeness 幅度明确；电商商品搜索完整性场景高度相关。综合评分 **78/100（STRONG）**。
📊 关键指标：多 BEIR 子集 completeness +36%（相对基线100 → 136）；端到端训练版本全部 BEIR 指标全优；多相关文档时差距持续扩大。

---

📄 标题：Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models
👥 作者：Logan Mann, Yi Xia, Ajit Saravanan, Ishan Dave et al. (UC Santa Barbara · Algoverse AI Research · UC Berkeley)
🔗 链接：https://arxiv.org/abs/2606.17389
📝 方法概述：VRP（VLM Reliability Probe）是一篇分析型论文，系统证伪"注意力聚焦⇒输出可靠"的主流假设。定义三类结构性注意力指标（簇数 Cₖ、空间熵 Hₛ、跨层演化 ΔHₛ），并以**自一致性**（同一问题多次采样推理路径的一致率）作为替代信号，跨 LLaVA、Qwen2-VL、PaliGemma 三族做系统实验。进一步通过因果层消融（主动破坏"最关键层"观察准确率崩溃）揭示不同架构可靠性存储位置的根本差异。
💡 创新性分析：跨三族大规模系统性证伪主流假设，相关性 R≈0.001 结论高度可信；因果层消融具有真正推断含义；"符号脱节/早锁定"现象解释幻觉产生机制；对内容置信度评估有直接参考价值。综合评分 **71/100（WEAK，分析型）**。
📊 关键指标：注意力-准确率相关 R≈0.001（近乎零）；自一致性-准确率相关 R=0.429（主导信号）；Qwen2-VL/PaliGemma 破坏 50% 关键层后高度鲁棒；LLaVA 相同操作后快速崩溃。

---

📄 标题：Looped World Models
👥 作者：Hongyuan Adam Lu, Z.L. Victor Wei et al. (FaceMind Research Asia)
🔗 链接：https://arxiv.org/abs/2606.18208
📝 方法概述：LoopWM（Looped World Models）是首个用于世界建模的循环参数共享架构。不堆叠 L 层深度网络，而是将**同一个参数共享的 Transformer 块**反复施加 K 次，迭代精炼潜在状态 s₀→s₁→…→sₖ。K 不固定：自适应计算模块根据每步预测难度动态分配循环次数，难步骤多循环、易步骤少循环。"迭代潜在深度"被确立为与模型参数量、训练数据正交的全新扩展轴。
💡 创新性分析：新扩展轴思路理论意义清晰；循环/参数共享用于世界模型首创；极端类别 0%→100% 表明强表达力；自适应计算比固定循环数设计更合理；与电商内容治理弱相关，作为高热点论文收录。综合评分 **70/100（WEAK，热点）**。
📊 关键指标：AlfWorld 综合得分 83%（最强基线 71%）；参数量压缩最高 100×（0.01×）；极端类别 Lifespan 准确率 0%→100%。
