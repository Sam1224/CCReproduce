## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer |
| **arXiv ID** | [2605.01732](https://arxiv.org/abs/2605.01732) |
| **提交日期** | 2026-05（May 2026） |
| **作者** | Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng |
| **机构** | Institute of Computing Technology, Chinese Academy of Sciences (ICT-CAS) |
| **领域** | Knowledge Distillation · LLM Compression · Token-Level Training |
| **Bucket** | WEAK |

---

## 方法概述 / Method Summary

EGAD 提出基于熵引导的自适应蒸馏策略，解决传统蒸馏对所有 token 一视同仁的低效问题：

**核心机制：**
1. **Token 级课程学习**：利用教师模型输出熵动态调整训练焦点，训练早期专注低熵（简单）token，逐步转向高熵（困难）token
2. **自适应温度调节**：蒸馏温度根据 token 熵动态调整，更好地捕获教师置信度模式
3. **双分支架构**：
   - 简单 token（低熵）：仅做 logits 蒸馏（轻量高效）
   - 困难 token（高熵）：追加深层特征蒸馏（更充分的知识迁移）

**理论基础：** 高熵 token 是模型决策最不确定的位置，也是最需要精细知识迁移的位置；低熵 token 浅层对齐即足够。

### Story Arc
> **传统 KD 对所有 token 等权处理，导致简单 token 浪费计算，困难 token 知识迁移不充分** → EGAD 用教师输出熵作为信号，通过课程学习+自适应温度+双分支架构实现 token 级精细蒸馏，在 LLM 压缩实验中验证有效性。

---

## 关键指标 / Key Metrics

| 实验设置 | 指标 | 说明 |
|--------|------|------|
| LLM 蒸馏（teacher→student） | Downstream Task Perf. | 优于 token 等权蒸馏 baseline |
| 推理效率 | Latency / Params | 压缩效果显著 |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 15/30 | 熵引导课程+双分支在 KD 中较新 |
| SOTA Delta | 8/15 | 验证了有效性但对比范围有限 |
| Exp Quality / Ablations | 9/15 | token 级分析有深度 |
| Efficiency | 7/10 | 减少无效计算 |
| Generalization | 3/5 | LLM 蒸馏场景 |
| Domain Relevance | 9/25 | 通用 LLM 压缩，适用内容审核模型部署 |
| **总分** | **51/100** | Feishu 推送 |
