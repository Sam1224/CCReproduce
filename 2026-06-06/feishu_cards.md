# Feishu Cards — 2026-06-06

---

📄 标题：UNIVID: Unified Vision-Language Model for Video Moderation
���� 作者：Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang et al.（字节跳动 ByteDance）
🔗 链接：https://arxiv.org/abs/2606.05748
📝 方法概述：UNIVID 是首个用单一视觉语言模型（VLM）全面替代大规模平台视频审核体系的工业级方案。模型生成"政策感知字幕"作为可解释中间表示，支撑三级级联审核流水线：（A）Risk Filter 融合多模态信号过滤高风险视频；（B）Moderation Actor 使用轻量版 UNIVID-Lite 和检索增强版 UNIVID-RAG 分别进行高吞吐审核决策与历史违规召回；（C）Trend Governance 对缓存字幕向量进行聚类，自适应检测新兴违规趋势。训练数据配方结合了专家人工精标与合成数据，解决了开源/商业 VLM 存在的安全护栏拒绝和政策对齐缺失问题。
💡 创新性分析：一是"字幕即中间表示"范式将黑盒分类器替换为可审计的自然语言决策依据，实现人工可验证的合规审核；二是 Trend Governance 模块利用字幕嵌入的聚类能力，无需重新训练即可响应新型违规趋势；三是工业部署中一举淘汰 1000+ 独立模型，大幅降低维护成本。
📊 关键指标：生产环境 A/B 测试——违规漏出率（Violation Leakage）相对降低 **42.7%**，过度审核率（Overkill Rate）相对降低 **37.0%**；替代 1000+ 专项模型，节省大量算力

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng（阿里巴巴国际数字商业集团）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：QueryAgent-R1 是一个记忆增强的强化学习 Agent，用于电商查询推荐（向用户主动推荐搜索词）。它将查询生成接地于真实商品检索（chain-of-retrieval），每一步 RL 训练都包含对真实商品库的检索调用，从而让 Agent 学会生成"检索结果能满足用户转化需求"的查询词。一致性奖励（Consistency Reward）同时优化查询点击率（CTR）和下游商品转化率（CVR），解决了现有方法"高 CTR 低 CVR"的困境。记忆模块存储用户历史查询与交互信息，支持个性化查询生成。
💡 创新性分析：核心创新在于将强化学习的奖励信号从"查询被点击"升级为"查询检索到的商品是否满足用户偏好"，通过在线检索闭环弥合了训练与部署的语义鸿沟。记忆增强与一致性奖励设计使 Agent 在生成阶段即完成对检索结果的自我评估与修正。
📊 关键指标：生产环境 A/B 测试（阿里巴巴国际平台）——查询 CTR **+2.9%**，引导转化率 Guided CVR **+3.1%**；离线公开数据集与自有工业数据集均持续优于强基线

---

📄 标题：OCL: Organizational Control Layer — Governance Infrastructure at the Execution Boundary of LLM Agent Systems
👥 作者：Tianyu Shi, Yang Mo, Yiou Liu et al.（McGill / Purdue / UNSW / UCLA / NYU / Stevens / Aimaikj Research）
🔗 链接：https://arxiv.org/abs/2606.04306
📝 方法概述：OCL 是一个与底层 LLM 模型无关的治理中间件，部署在 LLM Agent 的"生成"与"执行"之间。OCL 拦截每一个待执行动作，通过策略检查器、约束执行器、升级引擎和确定性重规划器对动作进行合规审查——违规动作将被确定性修正（而非重新提示 LLM），极大降低了非法执行率。在多 Agent 买卖博弈场景（AgenticPay 基准）中进行评估，测试了正常合规、对抗性操纵和提示注入三类场景。
💡 创新性分析：将安全保障从"LLM 上下文中的软约束"（system prompt 指令）升级为"执行边界处的硬约束"（不可绕过的中间件拦截），架构上的这一分离使治理机制对 LLM 提示注入攻击具有更强鲁棒性。无需修改底层模型即可适配任意 LLM 后端。
📊 关键指标：AgenticPay 基准（GPT-5.4 后端）——非法执行率从 **88% 降至接近零**，合规成功率从 **12% 升至 96%**；对抗拦截率：GPT-5.4 **94%**、Gemini-3.1 **82%**、Qwen-3.5 **60%**
