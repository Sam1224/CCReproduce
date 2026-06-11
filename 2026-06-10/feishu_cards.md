# Feishu Cards — 2026-06-10 (GMT+8)

每篇 score ≥ 40 的论文对应一张飞书卡片。

---

📄 标题：DocTrace: 基于多智能体的长文档QA可追踪证据选择框架
👥 作者：Zaixu Zhang, Xueguang Ma et al.（UNSW Sydney, CSIRO, 卧龙岗大学）
🔗 链接：https://arxiv.org/abs/2606.10921
📝 方法概述：将长文档QA任务拆分为检索、证据筛选、聚合、回答四个Agent角色，系统显式记录每步证据流转与选择理由，实现可审计的证据链。多Agent分工既提升了回答质量，又将推理成本降低约53%。
💡 创新性分析：首次把"证据链追踪"作为一等公民设计，对内容治理/审核链路中理由追溯有直接迁移价值；同时兼顾效果与成本是关键亮点。
📊 关键指标：NarrativeQA F1=40.28（vs ComoRAG 31.43）；多任务推理成本↓53.32%。评分：81/100。

---

📄 标题：SuperFashion: 基于超像素Token的服饰细粒度属性检索
👥 作者：Yaozong Lin, Lingxiao Meng et al.（中科院信工所, 中科院大学）
🔗 链接：https://arxiv.org/abs/2606.10697
📝 方法概述：用超像素（semantically coherent region）替代固定grid patch token，配合属性条件机制，使模型在检索时聚焦目标属性相关区域（如袖型、领型、图案），解决跨区域混叠问题。
💡 创新性分析：超像素token与服饰边界对齐动机明确，属性条件化检索设计完整，对电商服饰搜索/商品理解有直接落地价值，且可推广至其他细粒度属性检索场景。
📊 关键指标：FashionAI mAP=72.46（vs patch-token 71.15）；DARN mAP=62.15（vs 56.88，提升约9.3%）。评分：81/100。

---

📄 标题：Latent Memory: 通过Latent压缩为多模态RAG构建高效记忆
👥 作者：Zhao Zhang et al.（新加坡国立大学）
🔗 链接：https://arxiv.org/abs/2606.10572
📝 方法概述：提出可训练的Latent压缩器，将检索到的多模态证据（文本+图像）压缩为少量固定长度latent token，再送入生成模型，实现token数量级下降的同时效果不降甚至略升。
💡 创新性分析：把"证据压缩"做成端到端可训练而非规则截断，固定长度latent接口易于与各类RAG系统对接；对内容审核/理解等需要大量证据的场景尤为适用。
📊 关键指标：2Wiki+MuSiQue OOD F1=28.0（vs BM25 26.2，token数~71 vs ~209）；WebQA F1=69.4。评分：81/100。

---

📄 标题：GenAIR: LLM生成用户原型用于序列推荐的Item表征学习
👥 作者：Santiago B. Velasco, Clement Clement et al.（香港中文大学, McGill大学, 同济大学）
🔗 链接：https://arxiv.org/abs/2606.11023
📝 方法概述：用LLM为每个item生成一组可解释的用户原型（archetypes），训练时通过用户历史行为映射到原型权重，得到动态的"因人而异"item表征，再与SASRec/BERT4Rec等序列推荐模型联合优化。
💡 创新性分析：把LLM的语义理解能力用于构造可控、可组合的原型空间，并用行为数据校准使原型与真实偏好对齐，相比静态文本embedding更贴近用户实际需求。
📊 关键指标：在多数据集上HR@10/NDCG@10相对强基线取得稳定提升。评分：79/100。

---

📄 标题：miniReranker: 通过视觉缓存复用实现极速多模态重排
👥 作者：Binyuan Hui, Zhou Hong et al.（东方理工大学, Netmind.ai, LMU慕尼黑）
🔗 链接：https://arxiv.org/abs/2606.10759
📝 方法概述：针对电商多模态重排中商品图像反复出现的特点，将图像侧中间表示预先缓存，在线重排时只计算轻量query条件融合模块，实现>99%运行时减少且效果几乎无损。
💡 创新性分析：将系统层面的"候选复用"转化为模型层面的可复用表示，是工程友好型创新；对于大规模电商搜索重排系统具有极高实用价值。
📊 关键指标：MMEB-v2效果保持率96.3%（70.3 vs baseline 73.0），运行时减少>99%（1.49M query测试）。评分：78/100。

---

📄 标题：STORM: 奖励引导的Beam Search用于多模态检索Query扩展
👥 作者：Chuo Liu, Nora Belrose et al.（Mila, 巴黎萨克雷大学）
🔗 链接：https://arxiv.org/abs/2606.10621
📝 方法概述：将query expansion视为reward-guided beam search问题，通过检索质量代理信号对候选扩展打分并在beam内选择最优序列，使扩写方向直接服务于检索指标而非语言流畅性。
💡 创新性分析：把生成与检索闭环是关键创新，对电商搜索中的query改写/扩展有直接参考价值，尤其适合处理长尾/模糊查询。
📊 关键指标：TREC-DL-20 nDCG@10=67.3（vs BM25 48.0）；多个out-of-domain设置下一致提升。评分：75/100。

---

📄 标题：ARM: 面向自回归多模态模型的离散视觉Tokenization
👥 作者：Haonian Wang et al.（复旦大学, ByteDance TikTok, ByteDance Seed）
🔗 链接：https://arxiv.org/abs/2606.11188
📝 方法概述：设计适合自回归训练的离散视觉码本，将图像映射为可预测的离散token序列，统一多模态理解与生成到同一自回归框架，来自复旦+字节跳动合作。
💡 创新性分析：离散视觉token与语言模型自回归范式对齐是未来MLLM的基础技术方向，对电商内容理解与生成（商品图说生成、视觉问答）具有潜在价值。
📊 关键指标：POPE 87.3、MMMU 40.2、GenEval 0.86、WISE 0.56。评分：75/100。

---

📄 标题：Infini Memory: 基于主题-文档图的LLM Agent长期记忆
👥 作者：Yujie Zhang et al.（无限智能AI, 清华大学, 上海交通大学）
🔗 链接：https://arxiv.org/abs/2606.10677
📝 方法概述：提出Topic-Document Graph将Agent历史记忆层级组织，主题层定位相关簇后文档层精检索，配合写回/合并/遗忘策略，解决向量库记忆检索噪声与扩展性问题。
💡 创新性分析：层级图结构记忆比扁平向量库更利于长期维护与可控遗忘，对治理场景中的多轮处置记忆管理有迁移价值。
📊 关键指标：MemoryAgentBench总体64.7%，Accurate Retrieval 81.2%。评分：74/100。

---

📄 标题：From Prompt to Purchase: 在开放网络上评估LLM品牌推荐
👥 作者：Daniel Klymentov et al.（Scrunch AI）
🔗 链接：https://arxiv.org/abs/2606.10907
📝 方法概述：构建面向开放网络品牌推荐的评测框架，分析不同提示/模型在品牌推荐准确性、覆盖度与偏置上的表现，并连接到真实用户搜索/购买行为的影响讨论。
💡 创新性分析：量化了LLM品牌推荐对消费决策的影响，为电商治理中的广告合规与推荐可操控性评估提供基础分析工具。
📊 关键指标：多模型/提示设置下品牌推荐命中与转化分析（垂类差异显著）。评分：64/100。
