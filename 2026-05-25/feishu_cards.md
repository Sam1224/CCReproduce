# Feishu Cards — 2026-05-25 Daily AI Paper Inspection

> Papers with score ≥ 40. Domain: e-commerce content ecosystem & influencer governance.

---

📄 标题：VINA: Video as Natural Augmentation — 统一AI生成图像与视频检测
👥 作者：Zhengcen Li, Chenyang Jiang, Liangxiu Su 等 | 哈尔滨工业大学（深圳）+ 鹏城实验室
🔗 链接：https://arxiv.org/abs/2605.21977
📝 方法概述：VINA将视频帧作为物理上天然合理的增强数据，联合训练图像和视频AIGC检测器；通过跨模态监督对比目标消除图像检测器用于视频帧时的跨模态差距（视频编解码、色彩转换等引入的分布偏移）。单一统一模型无需数据集特定调优即可同时检测AI生成图像和视频。
💡 创新性分析：首创"视频作为自然增强"范式，颠覆以往图像/视频分别建模的惯例；跨模态对比学习将两种模态对齐到统一语义空间；在14个多样化基准上全面取得SOTA，无需复杂手工增强。
📊 关键指标：14个图像、视频及野外基准（AUC/ACC） — 全面SOTA；较图像/视频分别训练检测器均有双向提升。

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard — 模式条件化LLM安全护栏
👥 作者：Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis | Fastino Labs
🔗 链接：https://arxiv.org/abs/2605.07982 | 代码：https://github.com/fastino-ai/GLiGuard
📝 方法概述：GLiGuard是0.3B双向编码器，通过将安全任务定义和标签语义编码为结构化Token Schema注入输入序列，实现单次非自回归前向推理同时完成提示安全、响应安全、14种危害类别、越狱策略等所有维度的分类，比解码器类guard模型快16倍。
💡 创新性分析：Schema条件化设计将任务描述作为输入前缀，使同一模型灵活覆盖任意新增安全类别；非自回归推理从根本消除自回归延迟；0.3B规模下F1仅落后最强基线1.7点，实现效率-精度最优平衡。
📊 关键指标：Prompt safety F1 — 与最强基线差距≤1.7点；Response safety F1 — 开源guard模型第二名；延迟 — 比解码器guard降低16倍。

---

📄 标题：TGQ-Former: 文本引导视觉表示学习用于鲁棒多模态电商推荐
👥 作者：Yufei Guo, Jing Ma, Tianlu Zhang, Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong, Jungong Han | 清华大学 + 京东
🔗 链接：https://arxiv.org/abs/2605.17366
📝 方法概述：TGQ-Former（文本引导Q-Former）以结构化商品元数据（标题/类目/属性）作为语义引导，驱动Q-Former从含噪声的商品图像视觉Token中选取与商品核心语义对齐的视觉Token，过滤促销水印、背景干扰等虚假视觉线索，提升I2I检索鲁棒性。
💡 创新性分析：精准识别电商图片噪声（促销叠加）对MLLM连接器的危害，将元数据从辅助信息提升为视觉特征提取的主动引导信号；工业背景强（清华+京东），直接在真实电商数据上验证。
📊 关键指标：JD.COM商品I2I检索（含促销噪声图像） — Recall@K显著提升 vs. 标准LLaVA连接器；干净图像场景保持基线水平。

---

📄 标题：Latent Space Probing for Adult Content Detection in Video Generative Models — 视频生成模型潜空间探测成人内容检测
👥 作者：(作者未完全公开)
🔗 链接：https://arxiv.org/abs/2605.00874
📝 方法概述：在视频扩散模型（CogVideoX）推理过程中拦截去噪U-Net产生的潜在空间特征，附加轻量级探测分类器（CNN/MLP）进行实时成人内容检测；生成过程内部介入，无需等待完整像素输出，实现实时拦截。构建了11,039个视频片段（5,086违规/5,953非违规）的大规模数据集。
💡 创新性分析：首次将探测分类器附加于视频扩散模型内部潜在表示，开创"生成时检测"新范式；计算开销远低于事后像素空间检测；大规模视频安全标注数据集是重要贡献。
📊 关键指标：11,039视频片段二分类（AUC/ACC） — 显著优于事后像素检测；延迟 — 实时介入无需完整渲染。

---

📄 标题：GRMCRec: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion — 图检索增强模态补全的鲁棒多模态推荐
👥 作者：Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He | 新加坡国立大学
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：GRMCR通过模态感知子图检索机制从用户-商品交互图中选取语义相关邻居，用图Transformer联合编码查询节点与检索子图来补全缺失模态特征；可学习稀疏路由码本将补全嵌入正则化为紧凑基向量，提升鲁棒性。
💡 创新性分析：将RAG核心思想迁移至模态补全是新颖探索；充分利用协作过滤图结构作为补全上下文信息源；稀疏路由码本正则化是有效的嵌入紧凑化方案。
📊 关键指标：Amazon Baby/Sports/Clothing/Beauty（模态缺失率20%-80%） — Recall@20/NDCG@20均优于FREEDOM/SLMRec等基线。

---

📄 标题：GranuRAG: From Scenes to Elements — Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG — 细粒度证据检索的可验证多模态RAG
👥 作者：Guanhua Chen, Chuyue Huang, Yutong Yao, Shudong Liu, Xueqing Song, Lidia S. Chao, Derek F. Wong | 澳门大学等
🔗 链接：https://arxiv.org/abs/2605.15019
📝 方法概述：GranuRAG将视觉场景分解为细粒度元素作为检索单元，通过多粒度跨模态对齐在元素级别检索证据，生成时对检索证据显式归因，提供可追溯、可验证的推理链。配套GranuVistaVQA基准（地标+元素级标注+多视角）。
💡 创新性分析：将RAG检索粒度从场景级细化到元素级，解决粒度不匹配导致的验证难题；提供归因可追溯性是重要补充。
📊 关键指标：GranuVistaVQA基准（准确率） — 优于粗粒度RAG基线；归因准确性 — 元素级可验证。

---

📄 标题：LLaVA-CKD: Bottom-Up Cascaded Knowledge Distillation for Vision-Language Models — 级联知识蒸馏用于视觉语言模型
👥 作者：Nikolaos Gkalelis, Vasileios Mezaris | (CERTH-ITI)
🔗 链接：https://arxiv.org/abs/2605.10641
📝 方法概述：LLaVA-CKD引入中间助手（Teaching Assistant）模型构建多级级联蒸馏链（教师→TA→学生），自底向上逐级传递知识，缩小每次蒸馏的容量差距，在LLaVA-KD框架基础上改进。
💡 创新性分析：级联蒸馏缓解容量差距是已知策略的合理延伸；应用于LLaVA VLM框架具有工程实践价值。
📊 关键指标：VQA/MMBench（准确率） — 优于LLaVA-KD直接蒸馏，证明中间TA桥接有效。

---

📄 标题：RePAIR: Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization — 无需错误分类体系的RAG改进
👥 作者：Gongbo Zhang, Yifan Peng, Chunhua Weng | 哥伦比亚大学
🔗 链接：https://arxiv.org/abs/2605.18772
📝 方法概述：RePAIR直接学习缺陷RAG响应到错误缓解行动计划的映射，绕过需要预先定义细粒度错误分类体系的依赖；通过响应-行动学习范式提升Agentic RAG的错误修正鲁棒性。
💡 创新性分析：避开分类体系对齐难题是实用创新；端到端响应→行动映射更灵活；适合错误类型多变的真实场景。
📊 关键指标：多个QA基准（EM/F1） — 优于基于分类体系的Agentic RAG基线。
