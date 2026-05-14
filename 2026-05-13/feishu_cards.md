# Feishu Cards — 2026-05-13 Daily AI Paper Inspection

---

📄 标题：A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance
👥 作者：Global E-Commerce Search Relevance Team, ByteDance（字节跳动）
🔗 链接：https://arxiv.org/abs/2605.05991
📝 方法概述：提出电商搜索相关性多智能体闭环框架，用 User Agent（发现差案例）、Annotator Agent（多路推理标注）、Optimizer Agent（根因分析+自动修复）三个 LLM 智能体替代人工角色，实现从差案例识别到模型修复的全自动化管线。
💡 创新性分析：首次将电商搜索相关性优化的完整人工闭环（标注→优化→评估）智能体化。LLM 标注精度超过人工 2.4%，标注成本降低 75.4%，系统持续自我进化无需人工介入。
📊 关键指标：内部电商数据集上，标注精度比人工提升 2.4%，标注成本降低 75.4%；搜索相关性指标闭环多轮持续提升。综合评分：81/100。

---

📄 标题：ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring
👥 作者：Deyi Ji, Junyu Lu, Xuanyi Liu 等（工业界团队）
🔗 链接：https://arxiv.org/abs/2605.02200
📝 方法概述：针对广告治理中策略频繁更新导致历史标签失效的问题，提出三阶段框架：策略播种（Policy Seeding）→ 对抗式标签修正（Prosecutor-Defender-Umpire 三智能体辩论）→ 潜在违规知识挖掘。通过智能体间辩论自动修正历史标签、发现隐性违规灰色地带。
💡 创新性分析：首次引入"检察官-辩护人-仲裁员"三智能体对抗辩论架构解决广告政策非平稳性问题。仅需极少量新政策金标样本即可实现策略自适应，大幅优于传统微调基线。
📊 关键指标：在工业广告数据集和公开数据集上均显著优于传统微调基线；Stage III（潜在知识挖掘）Historical Recall 达峰；三阶段消融验证各部分贡献。综合评分：80/100。

---

📄 标题：Valley3: Scaling Omni Foundation Models for E-commerce
👥 作者：Zeyu Chen, Guanghao Zhou, Qixiang Yin 等（字节跳动 Valley 团队）
🔗 链接：https://arxiv.org/abs/2605.01278
📝 方法概述：首个面向全球电商的全模态（Text+Image+Video+Audio）大语言模型，基于 Qwen3-VL+Qwen3-Omni 扩展，通过四阶段持续预训练（音频对齐→跨模态指令跟随→电商知识注入→长上下文推理）构建，支持主动搜索工具调用（Agentic Search）。
💡 创新性分析：填补了电商场景全模态 MLLM 空白，原生支持直播音频理解。构建业界首个全模态电商评测基准（6 任务：商品理解/售后体验/搜索推荐/直播分析/短视频理解/内容审核治理）。
📊 关键指标：Omni E-commerce Benchmark 6 项任务全部优于强基线，同时通用域基准保持竞争力（未退化）。综合评分：76/100。

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：Fastino Labs 团队
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：基于 GLiNER2 适配的 0.3B 参数双向编码器安全护栏模型，Schema 条件化设计在单次前向传播中同时完成 Prompt/Response 安全性判断、14 类危害分类、11 种越狱策略识别，通过硬决策规则组合输出最终裁决。
💡 创新性分析：将 LLM 安全护栏从自回归生成式分类转向双向编码器多任务分类，实现 16x 推理延迟下降和 23–90x 参数量压缩，在小模型下接近最强解码器基线性能。完全开源。
📊 关键指标：Prompt 安全基准平均 F1=87.7（比最强基线仅差 1.7 F1），Response 安全基准排名开源护栏模型第二；推理延迟比 LlamaGuard4-12B 降低 16x。综合评分：74/100。
